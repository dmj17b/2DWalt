"""HTML Renderer for policy visualization during training.

This module provides functionality to render policy trajectories as HTML
visualizations that can be logged to WandB for remote monitoring.
"""

import os
import sys
from pathlib import Path
from typing import Callable, Tuple, List, Optional, Union

import brax
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.base import State
from brax.io import html, mjcf
import imageio

class HTMLRenderer:
    """Renders policy trajectories as interactive HTML visualizations.
    
    This renderer unrolls policy trajectories in a MuJoCo environment and
    converts them to HTML files that can be viewed in a browser or logged
    to WandB for remote monitoring.
    """
    
    def __init__(
        self,
        env,
        render_dir: Optional[str] = None,
        episode_length: Optional[int] = None,
    ):
        """Initialize the HTML renderer.
        
        Args:
            env: Brax environment with MuJoCo backend.
            render_dir: Directory to save HTML files. Defaults to './visualizations'.
            episode_length: Length of episodes to render. Defaults to env's episode length.
        """
        self.env = env
        self.mj_model = env._mj_model
        self.dt = env.dt
        
        # Create Brax System for HTML rendering
        sys = mjcf.load_model(self.mj_model)
        self.sys = sys.tree_replace({'opt.timestep': env.dt})
        
        # Set up rendering directory
        self.render_dir = render_dir or "./visualizations"
        Path(self.render_dir).mkdir(parents=True, exist_ok=True)
        
        # Set episode length
        if hasattr(env, 'max_episode_length'):
            self.episode_length = env.max_episode_length
        elif hasattr(env.config, 'episode_length'):
            self.episode_length = env.config.episode_length
        else:
            self.episode_length = episode_length or 1000
    
    def unroll_policy_trajectory(
        self,
        state: State,
        policy,
        key: jax.Array,
        num_steps: Optional[int] = None,
    ) -> Tuple[State, Tuple[jax.Array, jax.Array, jax.Array]]:
        """Unroll a policy trajectory for a given number of steps.
        
        Args:
            state: Initial environment state.
            policy: Policy function that takes (obs, key) and returns (actions, policy_data).
            key: JAX random key.
            num_steps: Number of steps to unroll. Defaults to episode_length.
        
        Returns:
            Tuple of (final_state, trajectory_data) where trajectory_data contains
            (qpos, xpos, xquat) arrays of shape (num_steps, ...).
        """
        if num_steps is None:
            num_steps = self.episode_length
        
        def step_fn(carry, unused_t):
            current_state, rng = carry
            rng, subkey = jax.random.split(rng)
            
            # Get action from policy
            actions, _ = policy(current_state.obs, subkey)
            
            # Step environment
            next_state = self.env.step(current_state, actions)
            
            # Collect trajectory data
            trajectory_data = (
                next_state.data.qpos,
                next_state.data.xpos,
                next_state.data.xquat,
            )
            
            return (next_state, rng), trajectory_data
        
        (final_state, _), trajectory = jax.lax.scan(
            step_fn,
            (state, key),
            None,
            length=num_steps,
        )
        
        return final_state, trajectory
    
    def _trajectory_to_states(
        self,
        trajectory: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> List[State]:
        """Convert trajectory data to a list of Brax State objects.
        
        Args:
            trajectory: Tuple of (qpos, xpos, xquat) arrays.
        
        Returns:
            List of State objects ready for rendering.
        """
        qpos, xpos, xquat = trajectory
        qpos = np.asarray(qpos)
        xpos = np.asarray(xpos)
        xquat = np.asarray(xquat)
        
        # Create base data structure
        data = mujoco.mjx.make_data(self.mj_model)
        contact = brax.mjx.pipeline._reformat_contact(
            self.sys, data.contact,
        )
        state_list = []
        num_steps = qpos.shape[0]
        
        for i in range(num_steps):
            # Create state from trajectory data
            # Note: xpos and xquat may include world body (index 0), skip if needed
            body_start = 1 if xpos.shape[-1] > len(xpos[i]) else 0
            
            state = State(
                q=qpos[i],
                qd=np.zeros(self.mj_model.nv),
                x = brax.base.Transform(pos = xpos[i][1:], rot = xquat[i][1:]),
                xd = brax.base.Motion(vel = np.zeros_like(data.cvel[1:, 3:]), ang = np.zeros_like(data.cvel[1:, :3])),
                contact = contact
            )
            state_list.append(state)
        
        return state_list
    # Add this method inside your HTMLRenderer class
    def render_trajectory_to_video(
        self,
        trajectory: Tuple[jax.Array, jax.Array, jax.Array],
        iteration: int,
        filename_prefix: str = "trajectory",
        camera: Union[int, str, mujoco.MjvCamera] = -1,

    ) -> str:
        """Render a trajectory to an MP4 video file natively via MuJoCo."""
        qpos, _, _ = trajectory
        qpos = np.asarray(qpos)
        
        # Initialize native MuJoCo data and renderer
        data = mujoco.MjData(self.mj_model)
        renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
        
        frames = []
        num_steps = qpos.shape[0]
        
        for i in range(num_steps):
            # Apply state and compute forward kinematics
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(self.mj_model, data)
            
            # Render frame
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
            
        renderer.close()
        
        # Save to MP4
        filename = f"{filename_prefix}_{iteration:06d}.mp4"
        filepath = os.path.join(self.render_dir, filename)
        
        # fps = 1 / dt (e.g., dt=0.02 -> 50fps)
        fps = int(1.0 / self.dt)
        imageio.mimsave(filepath, frames, fps=fps)
        
        return filepath
    def render_trajectory_to_html(
        self,
        trajectory: Tuple[jax.Array, jax.Array, jax.Array],
        iteration: int,
        filename_prefix: str = "trajectory",
    ) -> str:
        """Render a trajectory to an HTML file.
        
        Args:
            trajectory: Tuple of (qpos, xpos, xquat) arrays from unroll_policy_trajectory.
            iteration: Iteration/step number for naming.
            filename_prefix: Prefix for the HTML filename.
        
        Returns:
            Path to the generated HTML file.
        """
        # Convert trajectory to state list
        state_list = self._trajectory_to_states(trajectory)
        
        # Render to HTML
        html_string = html.render(
            sys=self.sys,
            states=state_list,
            height="100vh",
            colab=False,
        )
        
        # Save to file
        filename = f"{filename_prefix}_{iteration:06d}.html"
        filepath = os.path.join(self.render_dir, filename)
        with open(filepath, "w") as f:
            f.write(html_string)
        
        return filepath
    
    def render_and_get_html(
        self,
        trajectory: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> str:
        """Render a trajectory and return the HTML string (for WandB logging).
        
        Args:
            trajectory: Tuple of (qpos, xpos, xquat) arrays from unroll_policy_trajectory.
        
        Returns:
            HTML string ready for logging to WandB.
        """
        state_list = self._trajectory_to_states(trajectory)
        
        html_string = html.render(
            sys=self.sys,
            states=state_list,
            height="100vh",
            colab=False,
        )
        
        return html_string

