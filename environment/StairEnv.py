import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from typing import Any, Dict, Optional, Union  # Import type hints for function signatures.
import jax  # Import JAX for numerical computing and random number generation.
import jax.numpy as jp  # Import JAX's numpy as jp for array operations.
from ml_collections import config_dict  # Import config_dict for configuration management.
from mujoco_playground._src.dm_control_suite import common  # Import common utilities for dm_control_suite.
from configs.env_config import (SimConfig, RewardConfig, CommandConfig)  # Import environment and reward configuration dataclasses.
import environment.BaseEnv as BaseEnv  # Import the base environment class to inherit from.

class StairEnv(BaseEnv.BaseEnv):
    """Stair terrain environment for the 2D Walt robot."""

    def __init__(
            self,
            sim_config: SimConfig = SimConfig(),
            reward_config: RewardConfig = RewardConfig(),
            command_config: CommandConfig = CommandConfig(),
    ):

        spawn1 = jp.array([-10.0, 0.0, 0.0])
        spawn2 = jp.array([-4.0, 0.0, 1.3])
        spawn3 = jp.array([1.0, 0.0, 0.0])
        spawn4 = jp.array([6.2, 0.0, 1.3])
        self.spawn_points = jp.stack([spawn1, spawn2, spawn3, spawn4], axis=0)  # Define spawn points for the robot on the stair terrain.
        # Stair terrain parameters:
        super().__init__(sim_config, reward_config, command_config)  # Initialize the base environment with the provided configurations.


    def _add_terrain(self):
        self.model_spec.add_stair_heightfield()

    def _reset_model_pos(self, rng) -> jax.Array:
        """Resets the model to an initial state. Between stair segments"""
        qpos = jp.zeros(self.mjx_model.nq)  # Initialize qpos to zeros.
        spawn_idx = jax.random.randint(rng, shape=(), minval=0, maxval=4)  # Randomly select one of the predefined spawn points.
        qpos = qpos.at[self.x_slide_qpos_addr].set(self.spawn_points[spawn_idx, 0])  # Set the x position to the selected spawn point's x coordinate.
        qpos = qpos.at[self.z_slide_qpos_addr].set(self.spawn_points[spawn_idx, 2])  # Set the z position to the selected spawn point's z coordinate
        return qpos
    
    def _reset_model_vel(self, rng) -> jax.Array:
        """Resets the model velocities to an initial state."""
        max_vel = 2.0
        qvel = jax.random.uniform(rng, minval = -max_vel, maxval = max_vel, shape=(self.mjx_model.nv))  # Initialize qvel to small random values to encourage exploration.
        return qvel
    

def main():
    env = StairEnv()  # Create an instance of the StairEnv environment.
    print("StairEnv environment created successfully.")

if __name__ == "__main__":
    main()  # Run the main function to test environment creation.