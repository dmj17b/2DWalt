import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
import mujoco
from brax.io import html, mjcf
from brax.io import model
import brax
import numpy as np
from mujoco import mjx
import jax
import jax.numpy as jp
from brax.base import State, System, Env
from typing import Sequence, Tuple, List


def policy_render_callback():
    pass

def policy_step(
    env: Env,
    state: State,
    policy,
    key,
    extra_fields: Sequence[str] = (),
) -> State:
    actions, policy_data = policy(state.obs, key)
    next_state = env.step(state, actions)
    return next_state 

class HTMLRenderer:
    def __init__(self, env):
        self.env = env
        self.mj_model = env._mj_model
        # Create Brax System for HTML rendering:
        sys = mjcf.load_model(self.mj_model)
        self.sys = sys.tree_replace({'opt.timestep': env.dt})
        self.dt = env.dt

    def render(self, state, filename="render.html"):
        # Create an HTML renderer for the current environment
        renderer = html.Renderer(self.mj_model)

        # Render the current state to an HTML file
        renderer.render(state.data, filename)



    def unroll_policy_trajectory(
        env: Env,
        state: State,
        policy,
        key,
        num_steps: int,
        extra_fields: Sequence[str] = (),
    ) -> Tuple[State, List[State]]:
        """Unrolls a trajectory by applying the policy to the environment."""
        @jax.jit
        def f(carry, unused_t):
            state, key = carry
            key, subkey = jax.random.split(key)
            state = policy_step(
                env,
                state,
                policy,
                key,
                extra_fields=extra_fields,
            )
            return (state, subkey), (state.data.qpos, state.data.xpos, state.data.xquat)

        (final_state, _), trajectory = jax.lax.scan(
            f,
            (state, key),
            (),
            length=num_steps,
        )

        return final_state, trajectory

    def _render_html(
        self,
        states: List[Tuple[jax.Array, jax.Array, jax.Array]],
        iteration: int,
    ) -> None:
        """ Render using Brax HTML renderer. """
        qpos, xpos, xquat = jax.tree.map(lambda x: x[:, 0, :], states)
        data = mujoco.mjx.make_data(self.mj_model)
        data_args = data.__dict__
        data_args['contact'] = brax.mjx.pipeline._reformat_contact(
            self.sys, data.contact,
        )
        state_list = []
        for i in range(self.render_episode_length):
            state_list.append(
                brax.mjx.base.State(
                    q=qpos[i],
                    qd=np.zeros(self.mj_model.nv),
                    x=brax.base.Transform(
                        pos=xpos[i][1:],
                        rot=xquat[i][1:],
                    ),
                    xd=brax.base.Motion(
                        vel=np.zeros_like(data.cvel[1:, 3:]),
                        ang=np.zeros_like(data.cvel[1:, :3]),
                    ),
                    **data_args,
                ),
            )

        html_string = html.render(
            sys=self.sys,
            states=state_list,
            height="100vh",
            colab=False,
        )

        html_string = html.render(
            sys=self.sys,
            states=state_list,
            height="100vh",
            colab=False,
        )

        filepath = os.path.join(self.filepath, f'{iteration}.html')
        self.current_filepath = filepath
        with open(filepath, "w") as f:
            f.writelines(html_string)

