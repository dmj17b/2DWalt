import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from typing import Any, Dict, Optional, Union  # Import type hints for function signatures.
import jax  # Import JAX for numerical computing and random number generation.
import jax.numpy as jp  # Import JAX's numpy as jp for array operations.
from ml_collections import config_dict  # Import config_dict for configuration management.
from mujoco_playground._src.dm_control_suite import common  # Import common utilities for dm_control_suite.
from configs.env_config import (SimConfig, RewardConfig, StairCommandConfig)  # Import environment and reward configuration dataclasses.
import environment.BaseEnv as BaseEnv  # Import the base environment class to inherit from.
from mujoco_playground._src import mjx_env  # Import custom environment base class.

class StairEnv(BaseEnv.BaseEnv):
    """Stair terrain environment for the 2D Walt robot."""

    def __init__(
            self,
            sim_config: SimConfig = SimConfig(),
            reward_config: RewardConfig = RewardConfig(),
            command_config: StairCommandConfig = StairCommandConfig(),
            challenge_level: int = 0,
    ):

        spawn1 = jp.array([-25.5, 0.0, 0.0])
        spawn2 = jp.array([-6.3, 0.0, 0.0])
        spawn3 = jp.array([9.5, 0.0, 0.0])
        spawn4 = jp.array([22.0, 0.0, 0.0])
        spawn5 = jp.array([31.7, 0.0, 0.0])
        self.challenge_level = challenge_level  # Initialize the challenge level for curriculum learning, which determines the starting position on the stairs.
    
        self.spawn_points = jp.stack([spawn1, spawn2, spawn3, spawn4, spawn5], axis=0)  # Define spawn points for the robot on the stair terrain.
        # Stair terrain parameters:
        super().__init__(sim_config, reward_config, command_config)  # Initialize the base environment with the provided configurations.

    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets the environment to an initial state."""
        rng, terrain_rng, pos_rng, vel_rng, command_rng = jax.random.split(rng, 5)

        command = self.sample_command(command_rng)  # Sample an initial command for the environment
        steps_until_cmd_change = jax.random.randint(command_rng, (), self.min_steps_per_command, self.max_steps_per_command + 1)  # Sample the number of steps until the next command change

        info = {
            "rng": rng,
            "command": command,
            "prev_action": jp.zeros(self.action_size),  # Initialize previous action to zeros
            "steps_since_cmd_change": jp.zeros(()),  # Counter for steps since last command change
            "steps_until_cmd_change": steps_until_cmd_change,  # Counter for steps until next command change
            "challenge_level": self.challenge_level,
        } 

        qpos = self._reset_model_pos(info)  # Reset the model's position based on the challenge level
        qvel = self._reset_model_vel(vel_rng)  # Reset the model's velocities
        mocap_pos = self._reset_terrain(terrain_rng)  # Randomize terrain by setting mocap bodies to new positions

        data = mjx_env.make_data(
            self.model,
            qpos=qpos,
            qvel=qvel,
            mocap_pos=mocap_pos, 
            impl = self._config.impl,
            naconmax=self._config.naconmax,
        )

        metrics = {
            "reward/task": jp.zeros(()),
            "reward/body_pitch": jp.zeros(()),
            "reward/low_torques": jp.zeros(()),
            "reward/vel_tracking": jp.zeros(()),
            "reward/body_z_vel": jp.zeros(()),
            "reward/body_pitch_vel": jp.zeros(()),
            "reward/action_smoothing": jp.zeros(()),
            "reward/pitchover_penalty": jp.zeros(()), # ADD THIS LINE
            "train/episode_reward": jp.zeros(()),
            "train/episode_reward_err": jp.zeros(()),
        }

        reward = jp.zeros(())  # Scalar reward
        done = jp.zeros(())  # Scalar done flag

        obs = self._get_obs(data, info)  # Get the initial observation

        return mjx_env.State(data, obs, reward, done, metrics, info)
    # Defines a forward step in the environment given the current state and action.
    # Also computes the resulting observation, reward, done flag, and metrics.
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        motor_rng, rng = jax.random.split(state.info["rng"])  # Split the RNG for motor noise and other randomness in the step
        motor_targets = self.calculate_motor_targets(state, action, motor_rng)  # Calculate motor targets based on the current state and action

        data = mjx_env.step(
            self.mjx_model,
            state.data,
            motor_targets,
            self.n_substeps,
        )
        
        obs = self._get_obs(data, state.info)  # Get the observation after the step

        reward = self._get_reward(data, action, state.info, state.metrics)  # Compute the reward
        new_info = self._maybe_update_cmd(state.info)  # Maybe update the command and reset the counter if needed
        new_info["prev_action"] = action  # Update the previous action in the info dictionary for use in the next step

        # Failure condition: if the robot's body pitch exceeds a certain threshold, the episode is done.
        done = jp.where(jp.abs(data.qpos[self.y_rot_qpos_addr]) > self.max_body_pitch, 1.0, 0.0)  # Check if body pitch exceeds threshold and set done flag accordingly

        # Success condition: if the robot has successfully climbed the stairs, the episode is done.
        done = jp.where(self.check_success(data), 1.0, done) 
        success_reward = jp.where(self.check_success(data), self.reward_config.success_bonus, 0.0)  # Get the success bonus if the success condition is met



        reward = reward + success_reward  # Combine the step reward with the done penalty
        
        return mjx_env.State(data, obs, reward, done, state.metrics, new_info)
    

    
    def _add_terrain(self):
        self.model_spec.add_stair_heightfield()

    def _reset_model_pos(self, info) -> jax.Array:
        """Resets the model to an initial state. Between stair segments"""
        qpos = jp.zeros(self.mjx_model.nq)  # Initialize qpos to zeros.
        spawn_idx = info["challenge_level"]  # Get the spawn index from the info dictionary
        qpos = qpos.at[self.x_slide_qpos_addr].set(self.spawn_points[spawn_idx, 0])  # Set the x position to the selected spawn point's x coordinate.
        qpos = qpos.at[self.z_slide_qpos_addr].set(self.spawn_points[spawn_idx, 2])  # Set the z position to the selected spawn point's z coordinate
        return qpos
    
    def _reset_model_vel(self, rng) -> jax.Array:
        """Resets the model velocities to an initial state."""
        max_vel = 2.0
        qvel = jax.random.uniform(rng, minval = -max_vel, maxval = max_vel, shape=(self.mjx_model.nv))  # Initialize qvel to small random values to encourage exploration.
        return qvel
    
    def check_success(self, data) -> bool:
        """Checks if the robot has successfully climbed the stairs."""
        # Define success as having an z position greater than a certain threshold (e.g., reaching the top of the stairs)
        success_threshold = 1.8  # This threshold can be adjusted based on the stair layout
        return data.qpos[self.z_slide_qpos_addr] > success_threshold
    
    def check_failure(self, data) -> bool:
        """Checks if the robot has fallen down."""
        # Define failure as having a body pitch greater than a certain threshold
        return jp.abs(data.qpos[self.y_rot_qpos_addr]) > self.max_body_pitch

def main():
    env = StairEnv()  # Create an instance of the StairEnv environment.
    print("StairEnv environment created successfully.")

if __name__ == "__main__":
    main()  # Run the main function to test environment creation.