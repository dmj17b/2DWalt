import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU for this test
import EnvWalt2D
import mujoco
import mujoco.viewer
from typing import Optional, Dict, Union
from mujoco import mjx
import time
import jax
import jax.numpy as jp
print(jax.devices())

def main():
    # Initialize the environment
    env = EnvWalt2D.EnvWalt2D()  # Create an instance of the EnvWalt2D environment
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
            state = state.replace(data = mjx.put_data(env.mj_model, mj_data, impl=env._config.impl))

            # Print rewards for debugging
            # print(f"Step: {n_steps}, Total Reward: {state.reward}")
            # print(f"Body Pitch Reward: {state.metrics['reward/body_pitch']:.3f}")
            print(f"Low Torques Reward: {state.metrics['reward/low_torques']:.6f}")
            # print(f"Velocity Tracking Reward: {state.metrics['reward/vel_tracking']:.3f}\n")
            # Print body vel:
            # Call inference function to get action from the PPO policy every 5 sim steps:
            if n_steps % 5 == 0:
                key, subkey = jax.random.split(key)
                # action = jax.random.uniform(subkey, shape=(env.mjx_model.nu,), minval=-1.0, maxval=1.0)
                action = jp.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])


            state = step_fn(state, action)  # Step the environment
            n_steps += 1

            elapsed = time.time()-start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)  # Sleep to maintain real-time simulation
            

if __name__ == "__main__":
    main()



