import EnvWalt2D
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jp
from typing import Optional, Dict, Union
import mediapy as media

env = EnvWalt2D.EnvWalt2D()  # Create an instance of the EnvWalt2D environment
key = jax.random.PRNGKey(0)  # Initialize a random key for JAX


jit_reset = jax.jit(env.reset)  # JIT compile the reset function
jit_step = jax.jit(env.step)  # JIT compile the step function

state = jit_reset(key)  # Reset the environment to get the initial state
rollout = [state]
for i in range(env.config.episode_length):
    state = jit_step(state, jp.array([0.5, 1.5, 2.5, 2.5, -0.5, 1.5, 2.5, 2.5]))  # Take a step in the environment with specified actions
    print(f"Step {i}, Reward: {state.reward}, Done: {state.done}")  # Print the reward and done flag at each step
    if i % 10 == 0:
        rollout.append(state)

frames = env.render(rollout)  # Render the rollout to get frames
media.write_video("walt2d_jit.mp4", frames, fps=1/env.config.sim_dt)  # Save the frames as a video file