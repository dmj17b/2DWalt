import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from typing import Any, Dict, Optional, Union  # Import type hints for function signatures.
import warnings  # Import warnings module (not used in this snippet).

import jax  # Import JAX for numerical computing and random number generation.
import jax.numpy as jp  # Import JAX's numpy as jp for array operations.
from ml_collections import config_dict  # Import config_dict for configuration management.
import mujoco  # Import mujoco for physics simulation.
from mujoco import mjx  # Import mjx, a JAX-based Mujoco wrapper.

from mujoco_playground._src import mjx_env  # Import custom environment base class.
from mujoco_playground._src import reward  # Import reward utilities (not used in this snippet).
from mujoco_playground._src.dm_control_suite import common  # Import common utilities for dm_control_suite.
from configs.env_config import (SimConfig, RewardConfig, CommandConfig)  # Import environment and reward configuration dataclasses.

import modeling.GenModel as GenModel
import environment.base_walt as base_walt  # Import the base environment class to inherit from.

class FlatEnv(base_walt.BaseEnv):
    """Flat terrain environment for the 2D Walt robot."""
    
    def __init__(
            self,
            sim_config: SimConfig = SimConfig(),
            reward_config: RewardConfig = RewardConfig(),
            command_config: CommandConfig = CommandConfig(),
    ):
        super().__init__(sim_config, reward_config, command_config)  # Initialize the base environment with the provided configurations.

    def _add_terrain(self):
        """Add flat terrain to the environment"""
        self.model_spec.add_groundplane()  # Add a flat ground plane to the model specification.

    def _reset_model_pos(self, rng) -> jax.Array:
        """Resets the model to an initial state. Since this is a flat environment,
        we can simply initialize at zero each time. """
        qpos = jp.zeros(self.mjx_model.nq)  # Initialize qpos to zeros.
        qpos = qpos.at[self.z_slide_qpos_addr].set(0.05)  # Set the z position to 0.5 to be above the flat ground.
        return qpos
    
    def _reset_model_vel(self, rng) -> jax.Array:
        """Resets the model velocities to an initial state."""
        max_vel = 2.0
        qvel = jax.random.uniform(rng, minval = -max_vel, maxval = max_vel, shape=(self.mjx_model.nv))  # Initialize qvel to small random values to encourage exploration.
        return qvel
    
def main():
    env = FlatEnv()  # Create an instance of the FlatEnv environment.
    print("FlatEnv environment created successfully.")

if __name__ == "__main__":
    main()  # Run the main function to test environment creation.