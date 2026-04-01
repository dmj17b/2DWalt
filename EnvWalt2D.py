from typing import Any, Dict, Optional, Union  # Import type hints for function signatures.
import warnings  # Import warnings module (not used in this snippet).

import jax  # Import JAX for numerical computing and random number generation.
import jax.numpy as jp  # Import JAX's numpy as jp for array operations.
from ml_collections import config_dict  # Import config_dict for configuration management.
import mujoco  # Import mujoco for physics simulation.
from mujoco import mjx  # Import mjx, a JAX-based Mujoco wrapper.
import numpy as np  # Import numpy (not used in this snippet).

from mujoco_playground._src import mjx_env  # Import custom environment base class.
from mujoco_playground._src import reward  # Import reward utilities (not used in this snippet).
from mujoco_playground._src.dm_control_suite import common  # Import common utilities for dm_control_suite.
from env_config import (EnvConfig, RewardConfig)  # Import environment and reward configuration dataclasses.

import GenModel

def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the EnvWalt2D environment."""
    return config_dict.create(
        ctrl_dt = 0.01,
        sim_dt = 0.002,
        episode_length = 1000,
        action_repeat = 1,
        impl = 'jax',
    )



class EnvWalt2D(mjx_env.MjxEnv):
    
    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str,int,list[any]]]] = None,
            reward_config: RewardConfig = RewardConfig(),
    ):
        super().__init__(config, config_overrides = config_overrides) # Initialize the base class with config

        model_spec = GenModel.GenModel()  # Create an instance of the model generator
        model_spec.add_scene()  # Add the scene to the model
        
        # Load configurations
        self.config = config  # Store the configuration
        self.reward_config = reward_config  # Store the reward configuration

        # Command parameters
        self.max_vel_command = 2.0  # Maximum velocity command for the environment

        # Action scaling factors for different joints:
        self.hip_action_scale = 1.5  # Scaling factor for hip joint actions
        self.knee_action_scale = 1.5  # Scaling factor for knee joint actions
        self.wheel_action_scale = 15.0  # Scaling factor for wheel joint actions



        self._mj_model = model_spec.spec.compile()  # Compile the model and store it
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)  # Convert to JAX-compatible model

        self.default_ctrl = jp.zeros(self.mjx_model.nu)  # Default control inputs

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

    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets the environment to an initial state."""
        rng, command_rng, = jax.random.split(rng)
        qpos = self._reset_model_pos()  # Reset the model's position
        qvel = jp.zeros(self.mjx_model.nv)  # Initialize velocities to zero

        data = mjx_env.make_data(
            self.mjx_model,
            qpos=qpos,
            qvel=qvel,
        )

        metrics = {
            "reward/body_pitch": jp.zeros(()),
            "reward/low_torques": jp.zeros(()),
            "reward/vel_tracking": jp.zeros(()),
            "reward/body_z_vel": jp.zeros(()),
            "reward/body_pitch_vel": jp.zeros(()),
            "train/episode_reward": jp.zeros(()),
            "train/episode_reward_err": jp.zeros(()),
        }

        command = self.sample_command(command_rng)  # Sample an initial command for the environment

        reward = jp.zeros(())  # Scalar reward
        done = jp.zeros(())  # Scalar done flag

        info = {
            "rng": rng,
            "command": command,
            } 

        obs = self._get_obs(data, info)  # Get the initial observation


        return mjx_env.State(data, obs, reward, done, metrics, info)
    
    # Defines a forward step in the environment given the current state and action.
    # Also computes the resulting observation, reward, done flag, and metrics.
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        f_hip_target = self.default_ctrl[self.f_hip_act_addr] + self.hip_action_scale * action[0]
        f_knee_target = state.data.qpos[self.f_knee_qpos_addr] + self.knee_action_scale * action[1]
        f_wheel1_target = self.default_ctrl[self.f_wheel1_act_addr] + self.wheel_action_scale * action[2]
        f_wheel2_target = self.default_ctrl[self.f_wheel2_act_addr] + self.wheel_action_scale * action[3]
        r_hip_target = self.default_ctrl[self.r_hip_act_addr] + self.hip_action_scale * action[4]
        r_knee_target = state.data.qpos[self.r_knee_qpos_addr] + self.knee_action_scale * action[5]
        r_wheel1_target = self.default_ctrl[self.r_wheel1_act_addr] + self.wheel_action_scale * action[6]
        r_wheel2_target = self.default_ctrl[self.r_wheel2_act_addr] + self.wheel_action_scale * action[7]


        motor_targets = jp.array([
            f_hip_target, f_knee_target, f_wheel1_target, f_wheel2_target,
            r_hip_target, r_knee_target, r_wheel1_target, r_wheel2_target
        ])

        data = mjx_env.step(
            self.mjx_model,
            state.data,
            motor_targets,
            self.n_substeps,
        )
        
        obs = self._get_obs(data, state.info)  # Get the observation after the step

        reward = self._get_reward(data, action, state.info, state.metrics)  # Compute the reward

        done = jp.float32(0)    # No terminal state

        
        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    # Calculates reward based on the current state and action.
    def _get_reward(self,
                    data: mjx.Data,
                    action: jax.Array,
                    info: Dict[str, Any],
                    metrics: dict[str, Any],
    ) -> jax.Array:
        # Penalty for deviating too far from zero body pitch:
        body_pitch = data.qpos[self.y_rot_qpos_addr]  # Get the pitch of the body
        body_pitch_penalty = -self.reward_config.body_pitch*jp.square(body_pitch)  # Quadratic penalty for body pitch angle, scaled by reward_config 

        body_pitch_vel = data.qvel[self.y_rot_qpos_addr]  # Get the angular velocity of the body pitch
        body_pitch_vel_penalty = -self.reward_config.body_pitch_vel*jp.square(body_pitch_vel)  # Quadratic penalty for body pitch velocity, scaled by reward_config

        # Penalty for body z-velocity change (encourages maintaining consistent height):
        z_vel = data.qvel[self.z_slide_qpos_addr]  # Get the vertical velocity
        z_vel_penalty = -self.reward_config.body_z_vel*jp.square(z_vel)  # Quadratic penalty for vertical velocity, scaled by reward_config

        # Penalty for body height dropping below a specified value (encourages maintaining height):
        z_height = data.qpos[self.z_slide_qpos_addr]
        height_penalty = jp.where(z_height < -0.1, self.reward_config.height_penalty, 0.0)  # Apply penalty if height is below threshold
        

        # Penalize large torques
        joint_torques = data.qfrc_actuator  # Get the actuator forces
        low_torques_reward = jp.sum(jp.square(joint_torques))*self.reward_config.low_torques  # Reward low torque usage
        fwd_vel = data.qvel[0]  # Get the forward velocity
        vel_tracking_reward = self.tracking_reward(info["command"], fwd_vel, sigma=0.5)*self.reward_config.vel_tracking  # Reward forward velocity

        # Total reward
        episode_reward = body_pitch_penalty + body_pitch_vel_penalty + z_vel_penalty + low_torques_reward + vel_tracking_reward


        metrics["reward/body_pitch"] = body_pitch_penalty
        metrics["reward/body_pitch_vel"] = body_pitch_vel_penalty
        metrics["reward/low_torques"] = low_torques_reward
        metrics["reward/vel_tracking"] = vel_tracking_reward
        metrics["reward/body_z_vel"] = z_vel_penalty
        metrics["train/episode_reward"] = episode_reward

        return episode_reward

    # Helper function to compute a tracking reward based on the error between desired and actual values.
    # Uses exponential kernel to convert error into a reward, with a scaling factor sigma.
    def tracking_reward(self, desired, actual, sigma=0.25):
        desired = jp.array(desired)
        actual = jp.array(actual)
        error = jp.square(desired - actual)
        return jp.exp(-error / sigma)


    def _reset_model_pos(self) -> jax.Array:
        """Resets the model to an initial state."""
        qpos = jp.zeros(self.mjx_model.nq)
        return qpos

    """Returns the observation from the environment as a JAX array."""
    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        vel_command = jp.array([info["command"]])  # Get the velocity command from the info dictionary
        body_pitch = jp.array([data.qpos[self.y_rot_qpos_addr]]) # Get the pitch of the body
        f_hip_pos = jp.array([data.qpos[3]])  # Get the position of the front hip
        f_knee_pos = jp.array([data.qpos[4]])%(2*jp.pi)  # Get the position of the front knee (modulo 2pi to handle wrapping)
        f_knee_vel = jp.array([data.qvel[4]])  # Get the velocity of the front knee
        r_hip_pos = jp.array([data.qpos[7]])  # Get the position of the rear hip
        r_knee_pos = jp.array([data.qpos[8]])%(2*jp.pi)  # Get the position of the rear knee (modulo 2pi to handle wrapping)
        r_knee_vel = jp.array([data.qvel[8]])  # Get the velocity of the rear knee
        f_wheel1_vel = jp.array([data.qvel[5]])  # Get the velocity of the front wheel 1
        f_wheel2_vel = jp.array([data.qvel[6]])  # Get the velocity of the front wheel 2
        r_wheel1_vel = jp.array([data.qvel[9]])  # Get the velocity of the rear wheel 1
        r_wheel2_vel = jp.array([data.qvel[10]])  # Get the velocity of the rear wheel 2
        obs = jp.concatenate([
            vel_command,
            body_pitch,
            f_hip_pos, f_knee_pos, f_knee_vel, r_hip_pos, r_knee_pos, r_knee_vel,
            f_wheel1_vel, f_wheel2_vel, r_wheel1_vel, r_wheel2_vel
        ])
        return obs
    
    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng, command_rng = jax.random.split(rng)
        command = jax.random.uniform(command_rng, minval=-self.max_vel_command, maxval=self.max_vel_command)
        return command

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
    env = EnvWalt2D()
    state = env.reset(rng=jax.random.PRNGKey(0))
    new_state = env.step(state, jp.array([0.5, 1.5, 0.0, 0.0, -0.5, 1.5, 0.0, 0.0]))

if __name__ == "__main__":
    main()