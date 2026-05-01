import io
import os
import base64
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

try:
    import imageio
except Exception:
    imageio = None

try:
    from brax.io import html as brax_html
    from brax.base import State as BraxState
except Exception:
    brax_html = None
    BraxState = None

try:
    import wandb
except Exception:
    wandb = None


class EvalVisualizer:
    """
    Utility to run short deterministic eval rollouts, render them (GIF or Brax HTML),
    and optionally log the result to WandB.

    Init args:
    - host_env: environment object used for host rollouts (must support reset(), step(action), and render()).
    - brax_sys: optional Brax System required for brax HTML rendering.
    - save_dir: optional directory to store generated artifacts.
    - render_interval: only render when step % render_interval == 0 (set 1 to always).
    - max_steps: maximum host rollout steps when making GIF.
    - fps: gif frames per second.
    - render_size: (w,h) requested when calling env.render(...) if supported.
    - wandb_run: optional wandb.run instance to log results to.
    - prefer_brax_html: if True and brax_sys + states are provided, prefer brax.io.html.render.
    """

    def __init__(
        self,
        host_env: Optional[Any] = None,
        brax_sys: Optional[Any] = None,
        save_dir: Optional[str] = None,
        render_interval: int = 10000,
        max_steps: int = 300,
        fps: int = 15,
        render_size: Tuple[int, int] = (480, 320),
        wandb_run: Optional[Any] = None,
        prefer_brax_html: bool = False,
    ):
        self.host_env = host_env
        self.brax_sys = brax_sys
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.render_interval = int(render_interval)
        self.max_steps = int(max_steps)
        self.fps = int(fps)
        self.render_size = tuple(render_size)
        self.wandb_run = wandb_run
        self.prefer_brax_html = prefer_brax_html

        if imageio is None:
            raise RuntimeError("imageio is required for GIF rendering. pip install imageio")

    # ------------------- internal helpers -------------------
    def _build_inference(self, make_policy: Callable[..., Any], params: Any):
        """Try to create an inference callable from make_policy and params."""
        # Common brax pattern: make_policy(params) -> inference_fn
        try:
            return make_policy(params)
        except TypeError:
            # Try other common signatures
            try:
                return make_policy(params, deterministic=True)
            except Exception:
                # final fallback: partial wrapper (may still fail later)
                from functools import partial
                try:
                    return partial(make_policy, deterministic=True)(params)
                except Exception as e:
                    raise RuntimeError(f"Could not construct inference fn: {e}")

    def _normalize_frame(self, frame) -> Optional[np.ndarray]:
        """Return uint8 HxWx3 or None on failure."""
        try:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.dtype != np.uint8:
                # assume float in [0,1]
                arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
            return arr
        except Exception:
            return None

    # ------------------- host rollout -> GIF -------------------
    def rollout_and_gif_bytes(self, make_policy: Callable[..., Any], params: Any, max_steps: Optional[int] = None) -> Optional[bytes]:
        """Run deterministic host rollout using host_env, return GIF bytes or None."""
        if self.host_env is None:
            return None
        max_steps = int(max_steps or self.max_steps)
        inference = self._build_inference(make_policy, params)

        # reset
        try:
            obs = self.host_env.reset()
        except Exception:
            # try zero-arg reset variants
            try:
                obs = self.host_env.reset()
            except Exception:
                return None

        frames = []
        for t in range(max_steps):
            # compute action
            try:
                action = inference(obs)
            except Exception:
                # try passing batched obs
                try:
                    action = inference(np.asarray(obs))
                except Exception:
                    break

            # collapse batch dim if present
            try:
                a = np.asarray(action)
                if a.ndim > 1 and a.shape[0] == 1:
                    a = a[0]
                action_to_step = a
            except Exception:
                action_to_step = action

            # step
            try:
                step_out = self.host_env.step(action_to_step)
            except Exception:
                # try alternate step signature (obs, reward, terminated, truncated, info)
                try:
                    step_out = self.host_env.step(action_to_step)
                except Exception:
                    break

            # unpack step result robustly
            if isinstance(step_out, tuple):
                if len(step_out) == 4:
                    obs, rew, done, info = step_out
                elif len(step_out) == 5:
                    obs, rew, terminated, truncated, info = step_out
                    done = terminated or truncated
                else:
                    # unexpected
                    obs = step_out[0]
                    done = False
            else:
                # not tuple-like
                break

            # render
            frame = None
            # try common render calls; ignore exceptions
            tries = [
                lambda: self.host_env.render(mode="rgb_array", width=self.render_size[0], height=self.render_size[1]),
                lambda: self.host_env.render(mode="rgb_array"),
                lambda: self.host_env.render(),
                lambda: getattr(self.host_env, "render_rgb", lambda: None)(),
            ]
            for fn in tries:
                try:
                    f = fn()
                except Exception:
                    f = None
                if f is not None:
                    frame = f
                    break

            if frame is None:
                # nothing to render -> abort
                break

            arr = self._normalize_frame(frame)
            if arr is None:
                break
            frames.append(arr)

            # done flag
            try:
                if isinstance(done, (list, tuple, np.ndarray)):
                    if np.any(done):
                        break
                elif bool(done):
                    break
            except Exception:
                pass

        if not frames:
            return None

        buf = io.BytesIO()
        imageio.mimsave(buf, frames, format="GIF", fps=self.fps)
        buf.seek(0)
        return buf.read()

    # ------------------- brax HTML rendering -------------------
    def brax_html_from_states(self, states: Sequence[Any], height: str = "100vh") -> Optional[str]:
        """
        Render Brax HTML string from a sequence of brax.State-like objects (list of brax.base.State).
        Requires brax_html to be importable and self.brax_sys set.
        """
        if brax_html is None or self.brax_sys is None:
            return None
        try:
            html_str = brax_html.render(sys=self.brax_sys, states=states, height=height, colab=False)
            return html_str
        except Exception:
            return None

    # ------------------- logging / saving -------------------
    def _log_html(self, html_str: str, step: Optional[int] = None, key: str = "eval/simulation") -> None:
        """Log HTML to WandB if wandb_run present, else save to file if save_dir set."""
        step_arg = int(step) if step is not None else None
        if self.wandb_run is not None and wandb is not None:
            try:
                self.wandb_run.log({key: wandb.Html(html_str)}, step=step_arg)
                return
            except Exception:
                pass
        if self.save_dir is not None:
            filename = f"eval_{int(time.time())}_{step_arg or 0}.html"
            path = self.save_dir / filename
            path.write_text(html_str)
            return
        # last resort: no-op

    def _log_gif_bytes(self, gif_bytes: bytes, step: Optional[int] = None, key: str = "eval/simulation") -> None:
        """Log GIF embedded as HTML (data URI) to WandB or save file."""
        b64 = base64.b64encode(gif_bytes).decode("ascii")
        html_str = f"<html><body><img src='data:image/gif;base64,{b64}'/></body></html>"
        self._log_html(html_str, step=step, key=key)

    # ------------------- public convenience -------------------
    def evaluate_and_log(
        self,
        step: int,
        make_policy: Callable[..., Any],
        params: Any,
        *,
        use_brax_states: Optional[Sequence[Any]] = None,
        brax_height: str = "100vh",
        force: bool = False,
    ) -> None:
        """
        If step % render_interval == 0 (or force True), produce a rendering and log it.
        If use_brax_states is provided and brax_sys available, brax HTML rendering is used.
        Otherwise host_env -> GIF path is used.
        """
        if (not force) and (self.render_interval > 0) and (int(step) % self.render_interval != 0):
            return

        # prefer brax html if states provided and available
        if use_brax_states is not None and self.brax_sys is not None and brax_html is not None:
            html_str = self.brax_html_from_states(use_brax_states, height=brax_height)
            if html_str:
                self._log_html(html_str, step=step)
                return

        # fallback to host gif
        gif_bytes = self.rollout_and_gif_bytes(make_policy, params, max_steps=self.max_steps)
        if gif_bytes:
            self._log_gif_bytes(gif_bytes, step=step)
            return

        # nothing rendered; silent
        return