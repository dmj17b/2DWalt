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



class BaseEnv(mjx_env.MjxEnv):
    
    def __init__(
            self,
            sim_config: SimConfig = SimConfig(),
            reward_config: RewardConfig = RewardConfig(),
            command_config: CommandConfig = CommandConfig(),
    ):
        super().__init__(config = sim_config) # Initialize the base class with config

        # Generate the model to be used in the environment:
        self.model_spec = GenModel.GenModel()  # Create an instance of the model generator
        self.model_spec.add_scene()  # Add the scene to the model
        self._add_terrain()  # Add terrain to the model based on the specified type in the configuration

        # Load configurations
        self.config = sim_config  # Store the configuration
        self.reward_config = reward_config  # Store the reward configuration
        self.command_config = command_config  # Store the command configuration

        # Command parameters
        self.max_vel_command = self.command_config.max_vel  # Maximum velocity command for the environment
        self.zero_probability = self.command_config.zero_cmd_prob  # Probability of sampling a zero velocity command
        self.min_steps_per_command = int(self.command_config.min_cmd_duration / self.config.sim_dt)  # Minimum number of steps to maintain a command before resampling
        self.max_steps_per_command = int(self.command_config.max_cmd_duration / self.config.sim_dt)  # Maximum number of steps to maintain a command before resampling


        # Action scaling factors for different joints:
        self.hip_action_scale = 1.5  # Scaling factor for hip joint actions
        self.knee_action_scale = 0.05  # Scaling factor for knee joint actions
        self.wheel_action_scale = 15.0  # Scaling factor for wheel joint actions

        # Termination condition parameters:
        self.max_body_pitch = 1.5  # Maximum allowed body pitch angle (in radians) before episode termination

        self._mj_model = self.model_spec.spec.compile()  # Compile the model and store it
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)  # Convert to JAX-compatible model

        self.default_ctrl = jp.zeros(self.mjx_model.nu)  # Default control inputs
    
        self._define_addresses()  # Define addresses for relevant model components (e.g., joints, actuators)
        


    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets the environment to an initial state."""
        rng, terrain_rng, pos_rng, vel_rng, command_rng = jax.random.split(rng, 5)
        qpos = self._reset_model_pos(pos_rng)  # Reset the model's position
        qvel = self._reset_model_vel(vel_rng)  # Reset the model's velocities
        mocap_pos = self._reset_terrain(terrain_rng)  # Randomize terrain by setting mocap bodies to new positions

        data = mjx_env.make_data(
            self.mjx_model,
            qpos=qpos,
            qvel=qvel,
            mocap_pos=mocap_pos 
        )

        metrics = {
            "reward/task": jp.zeros(()),
            "reward/body_pitch": jp.zeros(()),
            "reward/low_torques": jp.zeros(()),
            "reward/vel_tracking": jp.zeros(()),
            "reward/body_z_vel": jp.zeros(()),
            "reward/body_pitch_vel": jp.zeros(()),
            "reward/action_smoothing": jp.zeros(()),
            "train/episode_reward": jp.zeros(()),
            "train/episode_reward_err": jp.zeros(()),
        }

        command = self.sample_command(command_rng)  # Sample an initial command for the environment
        steps_until_cmd_change = jax.random.randint(command_rng, (), self.min_steps_per_command, self.max_steps_per_command + 1)  # Sample the number of steps until the next command change


        reward = jp.zeros(())  # Scalar reward
        done = jp.zeros(())  # Scalar done flag

        info = {
            "rng": rng,
            "command": command,
            "prev_action": jp.zeros(self.action_size),  # Initialize previous action to zeros
            "steps_since_cmd_change": jp.zeros(()),  # Counter for steps since last command change
            "steps_until_cmd_change": steps_until_cmd_change,  # Counter for steps until next command change
            "knee_des_pos": jp.array([0.0, 0.0]),  # Desired knee positions 
            } 

        obs = self._get_obs(data, info)  # Get the initial observation


        return mjx_env.State(data, obs, reward, done, metrics, info)
    
    # Defines a forward step in the environment given the current state and action.
    # Also computes the resulting observation, reward, done flag, and metrics.
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        motor_targets = self.calculate_motor_targets(state, action)  # Calculate motor targets based on the current state and action

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
        new_info["knee_des_pos"] = jp.array([motor_targets[self.f_knee_act_addr], motor_targets[self.r_knee_act_addr]])  # Update the desired knee positions in the info dictionary for use in the next step

        # End episode if body pitch exceeds a certain threshold (encourages the robot to stay upright):
        done = jp.where(jp.abs(data.qpos[self.y_rot_qpos_addr]) > self.max_body_pitch, 1.0, 0.0)  # Check if body pitch exceeds threshold and set done flag accordingly
        done_penalty = -self.reward_config.terminal_pitch * done  # Apply a penalty to the reward if the episode is done due to excessive body pitch
        reward = reward + done_penalty  # Combine the step reward with the done penalty

        
        return mjx_env.State(data, obs, reward, done, state.metrics, new_info)
    
    def calculate_motor_targets(self, state: mjx_env.State, action: jax.Array) -> jax.Array:
        f_hip_target = self.default_ctrl[self.f_hip_act_addr] + self.hip_action_scale * action[0]
        f_knee_target = state.info["knee_des_pos"][0] + self.knee_action_scale * action[1]
        f_wheel1_target = self.default_ctrl[self.f_wheel1_act_addr] + self.wheel_action_scale * action[2]
        f_wheel2_target = self.default_ctrl[self.f_wheel2_act_addr] + self.wheel_action_scale * action[3]
        r_hip_target = self.default_ctrl[self.r_hip_act_addr] + self.hip_action_scale * action[4]
        r_knee_target = state.info["knee_des_pos"][1] + self.knee_action_scale * action[5]
        r_wheel1_target = self.default_ctrl[self.r_wheel1_act_addr] + self.wheel_action_scale * action[6]
        r_wheel2_target = self.default_ctrl[self.r_wheel2_act_addr] + self.wheel_action_scale * action[7]


        motor_targets = jp.array([
            f_hip_target, f_knee_target, f_wheel1_target, f_wheel2_target,
            r_hip_target, r_knee_target, r_wheel1_target, r_wheel2_target
        ])

        return motor_targets

    # Calculates reward based on the current state and action.
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

        # Penalty for body height dropping below a specified value (encourages maintaining height):
        z_height = data.qpos[self.z_slide_qpos_addr]
        height_penalty = jp.where(z_height < -0.1, -self.reward_config.height_penalty, 0.0)  # Apply penalty if height is below threshold
        

        # Penalize work
        joint_torques = data.qfrc_actuator[3:]  # Get the actuator forces
        low_torques_reward = -jp.sum(jp.square(joint_torques))*self.reward_config.low_torques  # Reward low torque usage


        # Action smoothing:
        action_smoothing = -jp.sum(jp.square(action - info["prev_action"])) * self.reward_config.action_smoothing

        # Total reward
        episode_reward = task_reward + body_pitch_vel_penalty + z_vel_penalty + low_torques_reward + action_smoothing

        metrics["reward/task"] = task_reward
        metrics["reward/body_pitch"] = body_pitch_penalty
        metrics["reward/body_pitch_vel"] = body_pitch_vel_penalty
        metrics["reward/low_torques"] = low_torques_reward
        metrics["reward/vel_tracking"] = vel_tracking_reward
        metrics["reward/body_z_vel"] = z_vel_penalty
        metrics["reward/action_smoothing"] = action_smoothing
        metrics["train/episode_reward"] = episode_reward

        return episode_reward

    # Helper function to compute a tracking reward based on the error between desired and actual values.
    # Uses exponential kernel to convert error into a reward, with a scaling factor sigma.
    def tracking_reward(self, desired, actual, sigma=0.25):
        desired = jp.array(desired)
        actual = jp.array(actual)
        error = jp.square(desired - actual)
        return jp.exp(-error / sigma)


    def _reset_model_pos(self, rng) -> jax.Array:
        """Resets the model to an initial state. Base - initialize zero each time."""
        qpos = jp.zeros(self.mjx_model.nq)
        return qpos
    
    def _reset_model_vel(self, rng) -> jax.Array:
        """Resets the model velocities to an initial state. Base - initialize zero each time."""
        qvel = jp.zeros(self.mjx_model.nv)  # Initialize qvel to zeros.
        return qvel
    
    def _reset_terrain(self, rng) -> jax.Array:
        """Resets the terrain to an initial state. Base - no terrain, so just return empty array."""
        return None


    """Returns the observation from the environment as a JAX array."""
    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:

        vel_command = jp.array([info["command"]])  # Get the velocity command from the info dictionary
        prev_action = info["prev_action"]  # Get the previous action from the info dictionary

        # Add noise to body pitch observation:
        pitch_noise_rng, rng = jax.random.split(info["rng"])
        noisy_pitch = data.qpos[self.y_rot_qpos_addr] + jax.random.normal(pitch_noise_rng)*0.01  # Add small noise to the body pitch observation to encourage robustness in the policy
        body_pitch_sin = jp.array([jp.sin(data.qpos[self.y_rot_qpos_addr])])  # Get the sine of the body pitch
        body_pitch_cos = jp.array([jp.cos(data.qpos[self.y_rot_qpos_addr])])  # Get the cosine of the body pitch


        # Add noise to joint positions observations:
        joint_noise_rng, rng = jax.random.split(rng)
        joint_pos_noise = jax.random.normal(joint_noise_rng, shape=(self.mjx_model.nq,)) * 0.05  # Add small noise to joint position observations to encourage robustness in the policy

        f_hip_pos = jp.array([data.qpos[self.f_hip_qpos_addr] + joint_pos_noise[self.f_hip_qpos_addr]])  # Add noise to the front hip position observation

        noisy_f_knee_pos = data.qpos[self.f_knee_qpos_addr] + joint_pos_noise[self.f_knee_qpos_addr]  # Add noise to the front knee position observation
        f_knee_sin = jp.array([jp.sin(noisy_f_knee_pos)])  # Use sine of knee angle to avoid discontinuities
        f_knee_cos = jp.array([jp.cos(noisy_f_knee_pos)])  # Include both sine and cosine to fully capture the knee angle information without discontinuities

        r_hip_pos = jp.array([data.qpos[self.r_hip_qpos_addr] + joint_pos_noise[self.r_hip_qpos_addr]])  # Add noise to the rear hip position observation

        noisy_r_knee_pos = data.qpos[self.r_knee_qpos_addr] + joint_pos_noise[self.r_knee_qpos_addr]  # Add noise to the rear knee position observation
        r_knee_sin = jp.array([jp.sin(noisy_r_knee_pos)])  # Use sine of knee angle to avoid discontinuities
        r_knee_cos = jp.array([jp.cos(noisy_r_knee_pos)])  # Include both sine and cosine to fully capture the knee angle information without discontinuities

        # Add noise to joint velocity observations:
        joint_vel_noise_rng, rng = jax.random.split(rng)
        joint_vel_noise = jax.random.normal(joint_vel_noise_rng, shape=(self.mjx_model.nv,)) * 0.5
        
        f_hip_vel = jp.array([data.qvel[self.f_hip_qpos_addr] + joint_vel_noise[self.f_hip_qpos_addr]])  # Get the velocity of the front hip
        r_hip_vel = jp.array([data.qvel[self.r_hip_qpos_addr] + joint_vel_noise[self.r_hip_qpos_addr]])  # Get the velocity of the rear hip

        f_knee_vel = jp.array([data.qvel[self.f_knee_qpos_addr] + joint_vel_noise[self.f_knee_qpos_addr]])  # Get the velocity of the front knee
        r_knee_vel = jp.array([data.qvel[self.r_knee_qpos_addr] + joint_vel_noise[self.r_knee_qpos_addr]])  # Get the velocity of the rear knee

        f_wheel1_vel = jp.array([data.qvel[self.f_wheel1_qpos_addr] + joint_vel_noise[self.f_wheel1_qpos_addr]])  # Get the velocity of the front wheel 1
        f_wheel2_vel = jp.array([data.qvel[self.f_wheel2_qpos_addr] + joint_vel_noise[self.f_wheel2_qpos_addr]])  # Get the velocity of the front wheel 2
        r_wheel1_vel = jp.array([data.qvel[self.r_wheel1_qpos_addr] + joint_vel_noise[self.r_wheel1_qpos_addr]])  # Get the velocity of the rear wheel 1
        r_wheel2_vel = jp.array([data.qvel[self.r_wheel2_qpos_addr] + joint_vel_noise[self.r_wheel2_qpos_addr]])  # Get the velocity of the rear wheel 2


        # Add noise to joint torque observations:
        torque_noise_rng, rng = jax.random.split(rng)
        torque_noise = jax.random.normal(torque_noise_rng, shape=(self.mjx_model.nu,)) * 0.05  # Add small noise to joint torque observations to encourage robustness
        joint_torques = data.qfrc_actuator[3:] + torque_noise  # Get the actuator torques for all joints
        obs = jp.concatenate([
            vel_command,
            body_pitch_sin, body_pitch_cos,
            f_hip_pos, f_hip_vel,
            f_knee_sin, f_knee_cos, f_knee_vel, 
            r_hip_pos, r_hip_vel,
            r_knee_sin, r_knee_cos, r_knee_vel,
            f_wheel1_vel, f_wheel2_vel, r_wheel1_vel, r_wheel2_vel,
            joint_torques,
            prev_action
        ])
        return obs
    

    def _maybe_update_cmd(self, info: dict[str, Any]) -> dict[str, Any]:
        """Checks if it's time to update the command and samples a new one if necessary."""
        new_info = dict(info)
        new_info["steps_since_cmd_change"] = info["steps_since_cmd_change"] + 1  # Increment the steps since last command change
        command_key, time_key, rng = jax.random.split(info["rng"], num=3)  # Split the RNG for command sampling and time sampling

        # Check if it's time to sample a new command:
        new_cmd = jp.where(
            new_info["steps_since_cmd_change"] >= new_info["steps_until_cmd_change"],
            self.sample_command(command_key),  # Sample a new command if the counter has reached the threshold
            info["command"]  # Otherwise, keep the current command
        )
        steps_since_cmd_change = jp.where(
            new_info["steps_since_cmd_change"] >= new_info["steps_until_cmd_change"],
            0,  # Reset the counter if a new command is sampled
            new_info["steps_since_cmd_change"]  # Otherwise, keep the current counter
        )
        steps_until_cmd_change = jp.where(
            new_info["steps_since_cmd_change"] >= new_info["steps_until_cmd_change"],
            jax.random.randint(time_key, (), self.min_steps_per_command, self.max_steps_per_command + 1),  # Sample a new duration for the next command if the counter has reached the threshold
            new_info["steps_until_cmd_change"]  # Otherwise, keep the current duration
        )
        new_info["steps_until_cmd_change"] = steps_until_cmd_change  # Update the steps until command change in the info dictionary
        new_info["command"] = new_cmd  # Update the command in the info dictionary
        new_info["steps_since_cmd_change"] = steps_since_cmd_change  # Update the counter in the info dictionary
        new_info["rng"] = rng  # Update the RNG in the info dictionary for the next step

        return new_info
    
    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng, command_rng, zero_rng = jax.random.split(rng, num=3)
        is_zero_command = jax.random.uniform(zero_rng) < self.zero_probability
        command = jax.random.uniform(command_rng, minval=-self.max_vel_command, maxval=self.max_vel_command)
        command = jp.where(is_zero_command, 0.0, command)
        return command
    
    def _define_addresses(self):
        # Define joint indices:
        self.x_slide_jid = self._mj_model.joint("x_slide").id
        self.z_slide_jid = self._mj_model.joint("z_slide").id
        self.y_rot_jid = self._mj_model.joint("y_rot").id
        self.f_hip_jid = self._mj_model.joint("front_hip").id
        self.f_knee_jid = self._mj_model.joint("front_knee").id
        self.f_wheel1_jid = self._mj_model.joint("front_wheel1").id
        self.f_wheel2_jid = self._mj_model.joint("front_wheel2").id
        self.r_hip_jid = self._mj_model.joint("rear_hip").id
        self.r_knee_jid = self._mj_model.joint("rear_knee").id
        self.r_wheel1_jid = self._mj_model.joint("rear_wheel1").id
        self.r_wheel2_jid = self._mj_model.joint("rear_wheel2").id

        # Define qpos addresses:
        self.x_slide_qpos_addr = self._mj_model.jnt_qposadr[self.x_slide_jid]
        self.z_slide_qpos_addr = self._mj_model.jnt_qposadr[self.z_slide_jid]
        self.y_rot_qpos_addr = self._mj_model.jnt_qposadr[self.y_rot_jid]
        self.f_hip_qpos_addr = self._mj_model.jnt_qposadr[self.f_hip_jid]
        self.f_knee_qpos_addr = self._mj_model.jnt_qposadr[self.f_knee_jid]
        self.f_wheel1_qpos_addr = self._mj_model.jnt_qposadr[self.f_wheel1_jid]
        self.f_wheel2_qpos_addr = self._mj_model.jnt_qposadr[self.f_wheel2_jid]
        self.r_hip_qpos_addr = self._mj_model.jnt_qposadr[self.r_hip_jid]
        self.r_knee_qpos_addr = self._mj_model.jnt_qposadr[self.r_knee_jid]
        self.r_wheel1_qpos_addr = self._mj_model.jnt_qposadr[self.r_wheel1_jid]
        self.r_wheel2_qpos_addr = self._mj_model.jnt_qposadr[self.r_wheel2_jid]
        

        # Define actuator addresses:
        self.f_hip_act_addr = self._mj_model.actuator("front_hip_act").id
        self.f_knee_act_addr = self._mj_model.actuator("front_knee_act").id
        self.f_wheel1_act_addr = self._mj_model.actuator("front_wheel1_act").id
        self.f_wheel2_act_addr = self._mj_model.actuator("front_wheel2_act").id
        self.r_hip_act_addr = self._mj_model.actuator("rear_hip_act").id
        self.r_knee_act_addr = self._mj_model.actuator("rear_knee_act").id
        self.r_wheel1_act_addr = self._mj_model.actuator("rear_wheel1_act").id
        self.r_wheel2_act_addr = self._mj_model.actuator("rear_wheel2_act").id

        # Define sensor addresses:
        self.body_vel_sensor_addr = self._mj_model.sensor("body_lin_vel").id

    def _add_terrain(self):
        raise NotImplementedError("Terrain not defined in base environment. Please implement terrain generation in a subclass based on the desired terrain type (e.g., flat, heightfield, boxes, stairs).")

    @property
    def xml_path(self) -> str:
        return "2DWalt.xml"
    
    @property
    def action_size(self) -> int:
        return self.mjx_model.nu
    
    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model
    
    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

def main():
    env = BaseEnv()
    state = env.reset(rng=jax.random.PRNGKey(0))
    new_state = env.step(state, jp.array([0.5, 1.5, 0.0, 0.0, -0.5, 1.5, 0.0, 0.0]))

if __name__ == "__main__":
    main()