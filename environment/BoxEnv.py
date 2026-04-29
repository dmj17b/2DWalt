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

class BoxEnv(BaseEnv.BaseEnv):
    """Box terrain environment for the 2D Walt robot."""

    def __init__(
            self,
            sim_config: SimConfig = SimConfig(),
            reward_config: RewardConfig = RewardConfig(),
            command_config: CommandConfig = CommandConfig(),
            difficulty: float = 0.5,  # Difficulty parameter to control box height and spacing in the terrain.
    ):
        self.difficulty = difficulty  # Difficulty level for terrain generation, can be used to scale box height and spacing.

        super().__init__(sim_config, reward_config, command_config)  # Initialize the base environment with the provided configurations.


    def _add_terrain(self):
        """Add box terrain to the environment"""
        # self.model_spec.add_groundplane()  # Add a flat ground plane to the model specification.
        self.model_spec.add_box_heightfield(spacing=64, difficulty=self.difficulty)  # Add a box heightfield to the model specification for terrain generation.

    def _reset_model_pos(self, rng) -> jax.Array:
        """Resets the model to an initial state. Between box obstacles"""
        qpos = jp.zeros(self.mjx_model.nq)  # Initialize qpos to zeros.
        x_pos = jax.random.uniform(rng, minval=-20.0, maxval=20.0)  # Sample a random x position for the robot within the specified range.
        qpos = qpos.at[self.z_slide_qpos_addr].set(0.05)  # Set the z position to 0.5 to be above the flat ground.
        qpos = qpos.at[self.x_slide_qpos_addr].set(x_pos)  # Set the x position to the sampled value.
        return qpos
    
    def _reset_model_vel(self, rng) -> jax.Array:
        """Resets the model velocities to an initial state."""
        max_vel = 2.0
        qvel = jax.random.uniform(rng, minval = -max_vel, maxval = max_vel, shape=(self.mjx_model.nv))  # Initialize qvel to small random values to encourage exploration.
        return qvel
    

def main():
    env = BoxEnv()  # Create an instance of the BoxEnv environment.
    print("BoxEnv environment created successfully.")

if __name__ == "__main__":
    main()  # Run the main function to test environment creation.