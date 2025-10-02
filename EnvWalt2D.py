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

import GenModel

def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the EnvWalt2D environment."""
    return config_dict.create(
        ctrl_dt = 0.01,
        sim_dt = 0.01,
        episode_length = 1000,
        action_repeat = 1,
        impl = 'jax',
        reward_config = config_dict.create(
            fwd_vel_weight = 1.0,
            body_pitch_weight = -0.5,
            low_torques_weight = -0.005,
            alive = 0.0,
            termination = -100.0,
        ),
    )



class EnvWalt2D(mjx_env.MjxEnv):
    
    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str,int,list[any]]]] = None,
    ):
        super().__init__(config, config_overrides = config_overrides) # Initialize the base class with config

        model_spec = GenModel.GenModel()  # Create an instance of the model generator
        model_spec.add_scene()  # Add the scene to the model
        
        self.config = config  # Store the configuration

        self._mj_model = model_spec.spec.compile()  # Compile the model and store it
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)  # Convert to JAX-compatible model
    

    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets the environment to an initial state."""
        rng, rng1 = jax.random.split(rng)
        qpos = self._reset_model_pos()  # Reset the model's position
        qvel = jp.zeros(self.mjx_model.nv)  # Initialize velocities to zero

        data = mjx_env.init(
            self.mjx_model,
            qpos=qpos,
            qvel=qvel,
        )

        metrics = {
            "reward/body_pitch": jp.zeros(()),
            "reward/low_torques": jp.zeros(()),
            "reward/fwd_vel": jp.zeros(()),
        }

        reward, done = jp.zeros(2)  # Initialize reward and done flag

        info = {"rng": rng}  # Store the RNG state in the info dictionary

        obs = self._get_obs(data, info)  # Get the initial observation


        return mjx_env.State(data, obs, reward, done, metrics, info)
    
    # Defines a forward step in the environment given the current state and action.
    # Also computes the resulting observation, reward, done flag, and metrics.
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(
            self.mjx_model,
            state.data,
            action,
            self.n_substeps,
        )
        
        obs = self._get_obs(data, state.info)  # Get the observation after the step

        reward = self._get_reward(data, action, state.info, state.metrics)  # Compute the reward

        done = jp.array(0.0)  # The episode is never done in this environment

        
        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    # Calculates reward based on the current state and action.
    def _get_reward(self,
                    data: mjx.Data,
                    action: jax.Array,
                    info: Dict[str, Any],
                    metrics: dict[str, Any],
    ) -> jax.Array:
        del info
        reward_weights = self._config.reward_config
        body_pitch = data.qpos[2]  # Get the pitch of the body
        body_pitch_reward = jp.pow(body_pitch, 2)*reward_weights.body_pitch_weight  # Reward low pitch angles
        joint_torques = data.qfrc_actuator  # Get the actuator forces
        low_torques_reward = jp.sum(joint_torques)*reward_weights.low_torques_weight  # Reward low torque usage
        fwd_vel = data.qvel[0]  # Get the forward velocity
        fwd_vel_reward = fwd_vel*reward_weights.fwd_vel_weight  # Reward forward velocity

        metrics["reward/body_pitch"] = body_pitch_reward
        metrics["reward/low_torques"] = low_torques_reward
        metrics["reward/fwd_vel"] = fwd_vel_reward

        return body_pitch_reward + low_torques_reward + fwd_vel_reward

    def _reset_model_pos(self) -> jax.Array:
        """Resets the model to an initial state."""
        qpos = jp.zeros(self.mjx_model.nq)
        return qpos

    """Returns the observation from the environment as a JAX array."""
    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info # Unused
        body_pos = jp.array([data.qpos[0]])  # Get the position of the body
        body_vel = data.qvel[0:2]  # Get the velocity of the body
        body_pitch = jp.array([data.qpos[2]]) # Get the pitch of the body
        f_hip_pos = jp.array([data.qpos[3]])  # Get the position of the front hip
        f_knee_pos = jp.array([data.qpos[4]])  # Get the position of the front knee
        f_knee_vel = jp.array([data.qvel[4]])  # Get the velocity of the front knee
        r_hip_pos = jp.array([data.qpos[7]])  # Get the position of the rear hip
        r_knee_pos = jp.array([data.qpos[8]])  # Get the position of the rear knee
        r_knee_vel = jp.array([data.qvel[8]])  # Get the velocity of the rear knee
        f_wheel1_vel = jp.array([data.qvel[5]])  # Get the velocity of the front wheel 1
        f_wheel2_vel = jp.array([data.qvel[6]])  # Get the velocity of the front wheel 2
        r_wheel1_vel = jp.array([data.qvel[9]])  # Get the velocity of the rear wheel 1
        r_wheel2_vel = jp.array([data.qvel[10]])  # Get the velocity of the rear wheel 2
        obs = jp.concatenate([
            body_pos, body_vel, body_pitch,
            f_hip_pos, f_knee_pos, f_knee_vel, r_hip_pos, r_knee_pos, r_knee_vel,
            f_wheel1_vel, f_wheel2_vel, r_wheel1_vel, r_wheel2_vel
        ])
        return obs

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