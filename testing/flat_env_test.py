import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU for this test
import environment.flat as FlatEnv  # Import the FlatEnv environment class to test environment creation.
import mujoco
import mujoco.viewer
from typing import Optional, Dict, Union
from mujoco import mjx
import time
import jax
import jax.numpy as jp
import pygame
from pygame import joystick
pygame.init()

print(jax.devices())

def main():
    # Initialize joystick for user control
    js = joystick.Joystick(0)  # Initialize the first joystick
    f_hip_pos = 0.0
    r_hip_pos = 0.0
    hip_delta = 0.05  # Increment for hip position command when D-pad is pressed

    # Initialize the environment
    env = FlatEnv.FlatEnv()  # Create an instance of the FlatEnv environment
    key = jax.random.PRNGKey(2)  # Initialize a random key for JAX

    # JIT compile the reset and step functions
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Reset the environment to get initial state
    key, subkey = jax.random.split(key)
    state = reset_fn(subkey)
    print(f"Commanded Velocity: {state.info['command']:.3f}")

    # Create standard CPU mj_data
    mj_data = mujoco.MjData(env._mj_model)

    # Pull the initial MJX state into standard CPU mj_data
    mjx.get_data_into(mj_data, env.mj_model, state.data)
    dt = env._mj_model.opt.timestep  # Get the simulation timestep
    n_steps = 0

    # Launch standard MuJoCo viewer
    with mujoco.viewer.launch_passive(env.mj_model, mj_data) as viewer:
        while viewer.is_running():
            # Keep track of step time
            start_time = time.time()
            # Update the standard CPU mj_data with the new MJX state
            mjx.get_data_into(mj_data, env.mj_model, state.data)

            viewer.sync()  # Sync the viewer to update the visualization

            # Update the MJX state with any changes from viewer interactions (e.g., user dragging the model)
            state = state.replace(
                data=state.data.replace(
                    qpos=jp.array(mj_data.qpos),
                    qvel=jp.array(mj_data.qvel),
                    qfrc_applied=jp.array(mj_data.qfrc_applied),
                    xfrc_applied=jp.array(mj_data.xfrc_applied),
                    ctrl=jp.array(mj_data.ctrl),
                )
            )

            # Call inference function to get action from the PPO policy every 5 sim steps:
            if n_steps % 5 == 0:
                pygame.event.pump()  # Process event queue to update joystick state
                ax1 = js.get_axis(0)  # Get the horizontal axis of the first joystick
                ax2 = js.get_axis(1)  # Get the vertical axis of the first joystick
                ax3 = js.get_axis(2)  # Get the horizontal axis of the second joystick
                ax4 = js.get_axis(3)  # Get the vertical axis of the second joystick
                d_pad_y = js.get_hat(0)[1]  # Get the vertical value of the D-pad
                d_pad_x = js.get_hat(0)[0]  # Get the horizontal value of the D-pad
                knee_vel = -ax2
                wheel_vel = -ax4
                f_hip_pos = d_pad_y * hip_delta + f_hip_pos
                r_hip_pos = d_pad_x * hip_delta + r_hip_pos
                action = jp.array([-f_hip_pos, knee_vel, wheel_vel, wheel_vel, r_hip_pos, knee_vel, wheel_vel, wheel_vel])





            state = step_fn(state, action)  # Step the environment

            n_steps += 1

            if n_steps > 2000:
                key, subkey = jax.random.split(key)
                state = reset_fn(subkey)  # Reset the environment after 2000 steps for testing purposes
                n_steps = 0

            elapsed = time.time()-start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)  # Sleep to maintain real-time simulation
            

if __name__ == "__main__":
    main()



