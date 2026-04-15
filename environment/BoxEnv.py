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
    ):
        # Define box terrain parameters
        self.n_boxes = 15  # Number of box obstacles in the environment
        self.box_x_range = 30.0  # Range of x positions for box placement
        self.box_width_range = (0.15, 1.0)  # Range of box widths
        self.box_max_height = 0.3  # Maximum height of the boxes to ensure they are small obstacles rather than walls

        super().__init__(sim_config, reward_config, command_config)  # Initialize the base environment with the provided configurations.
        self.mocap_ids = self._get_mocap_ids()  # Get the mocap body IDs for the box obstacles to control their positions during terrain randomization

        

    def _add_terrain(self):
        """Add box terrain to the environment"""
        self.model_spec.add_groundplane()  # Add a flat ground plane to the model specification.
        self.model_spec.add_box_obstacles(n_boxes = self.n_boxes,
                                          x_range = self.box_x_range,
                                          width_range = self.box_width_range,
                                          max_height = self.box_max_height
                                          )  # Add box obstacles to the environment for testing purposes.

    def _reset_model_pos(self, rng) -> jax.Array:
        """Resets the model to an initial state. Between box obstacles"""
        qpos = jp.zeros(self.mjx_model.nq)  # Initialize qpos to zeros.
        # Possible spawn locations are between the box obstacles, so we can sample from a set of discrete positions that are evenly spaced between the boxes.
        possible_x_positions = jp.linspace(-self.box_x_range, self.box_x_range, self.n_boxes)
        x_pos = jax.random.choice(rng, possible_x_positions[3:-3]) + (self.box_x_range/self.n_boxes)  # Sample a random x position from the possible positions.
        qpos = qpos.at[self.z_slide_qpos_addr].set(0.05)  # Set the z position to 0.5 to be above the flat ground.
        qpos = qpos.at[self.x_slide_qpos_addr].set(x_pos)  # Set the x position to the sampled value.
        return qpos
    
    def _reset_model_vel(self, rng) -> jax.Array:
        """Resets the model velocities to an initial state."""
        max_vel = 2.0
        qvel = jax.random.uniform(rng, minval = -max_vel, maxval = max_vel, shape=(self.mjx_model.nv))  # Initialize qvel to small random values to encourage exploration.
        return qvel
    
    def _reset_terrain(self, rng) -> jax.Array:
        """ Randomized box positions and heights for terrain randomization. """
        
        # 1. Generate the 1D arrays for X, Y, and Z coordinates
        box_heights = jax.random.uniform(rng, 
                                        minval=-self.box_max_height, 
                                        maxval=self.box_max_height/2, 
                                        shape=(self.n_boxes,)) 
        
        box_x_positions = jp.linspace(-self.box_x_range, 
                                    self.box_x_range, 
                                    self.n_boxes)
        
        # Y is always zero, but we need an array of the same shape
        box_y_positions = jp.zeros((self.n_boxes,))
        
        # 2. Stack them together. 
        # axis=1 turns three (N,) arrays into one (N, 3) matrix.
        new_box_positions = jp.stack([box_x_positions, box_y_positions, box_heights], axis=1)
        
        # 3. Initialize the base mocap array
        mocap_pos = jp.zeros((self.mjx_model.nmocap, 3))
        
        # 4. Vectorized Assignment (No loop needed!)
        # JAX maps the N rows of new_box_positions directly to the N indices in self.mocap_ids
        mocap_pos = mocap_pos.at[jp.array(self.mocap_ids)].set(new_box_positions)
        
        return mocap_pos

    def _get_mocap_ids(self):
        """Get the mocap body IDs for the terrain boxes."""
        mocap_ids = []
        for i in range(self.n_boxes):
            box_body_id = self.mj_model.body(f"box_{i}").id  # Get the body ID for the box obstacle
            mocap_id = self.mj_model.body_mocapid[box_body_id]  # Get the corresponding mocap ID for the box body
            mocap_ids.append(mocap_id)
        return jp.array(mocap_ids)

def main():
    env = BoxEnv()  # Create an instance of the BoxEnv environment.
    print("BoxEnv environment created successfully.")

if __name__ == "__main__":
    main()  # Run the main function to test environment creation.