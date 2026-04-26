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

class HFieldEnv(BaseEnv.BaseEnv):
    """Heightfield terrain environment for the 2D Walt robot."""

    def __init__(
            self,
            sim_config: SimConfig = SimConfig(),
            reward_config: RewardConfig = RewardConfig(),
            command_config: CommandConfig = CommandConfig(),
            difficulty: float = 0.0,  # Difficulty parameter to control terrain roughness and obstacle placement.
    ):
        # Define terrain parameters for the heightfield
        self.difficulty = difficulty  # Difficulty level for terrain generation, can be used to scale height and obstacle placement.

        super().__init__(sim_config, reward_config, command_config)  # Initialize the base environment with the provided configurations.
        self.mocap_ids = self._get_mocap_ids()  # Get the mocap body IDs for the box obstacles to control their positions during terrain randomization

        

    def _add_terrain(self):
        """Add heightfield terrain to the environment"""
        self.model_spec.add_hfield(height = self.difficulty * 0.5 + 0.5,  # Scale height by difficulty to create rougher terrain at higher difficulty levels.
                                   sigma = 0.6) 

    def _reset_model_pos(self, rng) -> jax.Array:
        """Resets the model to an initial state. Between box obstacles"""
        qpos = jp.zeros(self.mjx_model.nq)  # Initialize qpos to zeros.
        x_pos = jax.random.uniform(rng, minval=-15, maxval=15)  # Sample a random x position for the robot within the terrain bounds.
        qpos = qpos.at[self.x_slide_qpos_addr].set(x_pos)  # Set the x position to the sampled value.
        return qpos
    
    def _reset_model_vel(self, rng) -> jax.Array:
        """Resets the model velocities to an initial state."""
        max_vel = 2.0
        qvel = jax.random.uniform(rng, minval = -max_vel, maxval = max_vel, shape=(self.mjx_model.nv))  # Initialize qvel to small random values to encourage exploration.
        return qvel
    
    def _reset_terrain(self, rng) -> jax.Array:
        """ Randomized box positions and heights for terrain randomization. """
        mocap_y_pos = jax.random.uniform(rng, minval = -15, maxval = 15)
        mocap_pos = jp.array([[0.0, mocap_y_pos, 0.0]])  # The mocap position for the heightfield terrain. We can randomize the y position to create different terrain configurations.
        
        return mocap_pos

    def _get_mocap_ids(self):
        """Get the mocap body ID for the terrain"""
        hfield_body_id = self.mj_model.body("terrain_body").id  # Get the body ID for the heightfield terrain.
        hfield_mocap_id = self.mj_model.body_mocapid[hfield_body_id]  # Get the mocap ID for the heightfield body to control its position during terrain randomization.
        return jp.array([hfield_mocap_id])  # Return the mocap ID as a JAX array for use in the environment.

def main():
    env = HFieldEnv()  # Create an instance of the HFieldEnv environment.
    print("HFieldEnv environment created successfully.")

if __name__ == "__main__":
    main()  # Run the main function to test environment creation.