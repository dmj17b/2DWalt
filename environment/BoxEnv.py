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
from mujoco import mjx  # Import mujoco's mjx module for working with Mujoco environments.


class BoxEnv(BaseEnv.BaseEnv):
    """Box terrain environment for the 2D Walt robot."""

    def __init__(
            self,
            sim_config: SimConfig = SimConfig(),
            reward_config: RewardConfig = RewardConfig(),
            command_config: CommandConfig = CommandConfig(),
            difficulty: float = 0.5,  # Difficulty parameter to control box height and spacing in the terrain.
            spacing: int = 64,  # Spacing parameter for the box terrain generation.
    ):
        # Box terrain parameters:
        self.spacing = spacing  # Spacing for the box terrain generation, can be used to control the density of boxes.
        self.difficulty = difficulty  # Difficulty level for terrain generation, can be used to scale box height and spacing.

        super().__init__(sim_config, reward_config, command_config)  # Initialize the base environment with the provided configurations.


    def _add_terrain(self):
        """Add box terrain to the environment"""
        # self.model_spec.add_groundplane()  # Add a flat ground plane to the model specification.
        self.model_spec.add_box_heightfield(spacing=self.spacing, difficulty=self.difficulty)  # Add a box heightfield to the model specification for terrain generation.

    def _reset_model_pos(self, rng) -> jax.Array:
        """Resets the model to an initial state. Between box obstacles"""
        qpos = jp.zeros(self.mjx_model.nq)  # Initialize qpos to zeros.
        x_pos = jax.random.uniform(rng, minval=-15.0, maxval=15.0)  # Sample a random x position for the robot within the specified range.
        qpos = qpos.at[self.z_slide_qpos_addr].set(0.05)  # Set the z position to 0.5 to be above the flat ground.
        qpos = qpos.at[self.x_slide_qpos_addr].set(x_pos)  # Set the x position to the sampled value.
        return qpos
    
    def _reset_model_vel(self, rng) -> jax.Array:
        """Resets the model velocities to an initial state."""
        max_vel = 2.0
        qvel = jax.random.uniform(rng, minval = -max_vel, maxval = max_vel, shape=(self.mjx_model.nv))  # Initialize qvel to small random values to encourage exploration.
        return qvel
    def _get_reward(self,
                    data: mjx.Data,
                    action: jax.Array,
                    info: Dict[str, Any],
                    metrics: dict[str, Any],
    ) -> jax.Array:
        # Reward for tracking the commanded velocity:
        body_frame_x_vel = data.sensordata[0]
        vel_tracking_reward = self.tracking_reward(info["command"], body_frame_x_vel, sigma=0.2)*self.reward_config.vel_tracking 


        # Penalty for deviating too far from zero body pitch:
        body_pitch = data.qpos[self.y_rot_qpos_addr]  # Get the pitch of the body
        body_pitch_penalty = jp.exp(-self.reward_config.body_pitch*jp.square(body_pitch))  # Alternative: Exponential penalty for body pitch angle, scaled by reward_config

        task_reward = vel_tracking_reward*body_pitch_penalty  # Combine velocity tracking reward with exponential body pitch penalty 


        body_pitch_vel = data.qvel[self.y_rot_qpos_addr]  # Get the angular velocity of the body pitch
        body_pitch_vel_penalty = -self.reward_config.body_pitch_vel*jp.square(body_pitch_vel)  # Quadratic penalty for body pitch velocity, scaled by reward_config

        # Penalty for body z-velocity change (encourages maintaining consistent height):
        z_vel = data.sensordata[2]  # Get the vertical velocity in body frame
        z_vel_penalty = -self.reward_config.body_z_vel*jp.square(z_vel)  # Quadratic penalty for vertical velocity, scaled by reward_config

        
        joint_vel_penalty = jp.where(jp.abs(info["command"]) < 0.1, -self.reward_config.joint_vel * jp.sum(jp.square(data.qvel)), 0.0)  # Apply joint velocity penalty only when the velocity command is close to zero

        # Penalize work
        joint_torques = data.qfrc_actuator[3:]  # Get the actuator forces
        low_torques_reward = -jp.sum(jp.square(joint_torques))*self.reward_config.low_torques  # Reward low torque usage


        # Action smoothing:
        action_smoothing = -jp.sum(jp.square(action - info["prev_action"])) * self.reward_config.action_smoothing

        # End episode if body pitch exceeds a certain threshold (encourages the robot to stay upright):
        done = jp.where(jp.abs(data.qpos[self.y_rot_qpos_addr]) > self.max_body_pitch, 1.0, 0.0)  # Check if body pitch exceeds threshold and set done flag accordingly
        done_penalty = -self.reward_config.terminal_pitch * done  # Apply a penalty to the reward if the episode is done due to excessive body pitch


        # Total reward
        episode_reward = task_reward + vel_tracking_reward + body_pitch_vel_penalty + low_torques_reward + action_smoothing + joint_vel_penalty

        metrics["reward/task"] = task_reward
        metrics["reward/body_pitch"] = body_pitch_penalty
        metrics["reward/body_pitch_vel"] = body_pitch_vel_penalty
        metrics["reward/low_torques"] = low_torques_reward
        metrics["reward/vel_tracking"] = vel_tracking_reward
        metrics["reward/body_z_vel"] = z_vel_penalty
        metrics["reward/action_smoothing"] = action_smoothing
        metrics["reward/pitchover_penalty"] = done_penalty
        metrics["train/episode_reward"] = episode_reward

        return episode_reward

def main():
    env = BoxEnv()  # Create an instance of the BoxEnv environment.
    print("BoxEnv environment created successfully.")

if __name__ == "__main__":
    main()  # Run the main function to test environment creation.