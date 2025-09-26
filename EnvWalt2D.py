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
    )



class EnvWalt2D(mjx_env.MjxEnv):
    
    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str,int,list[any]]]] = None,
    ):
        super().__init__(config, config_overrides = config_overrides) # Initialize the base class with config

        model_spec = GenModel()  # Create an instance of the model generator
        model_spec.add_scene()  # Add the scene to the model
        
        self.config = config  # Store the configuration

        self._mj_model = model_spec.compile()  # Compile the model and store it
        self.mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)  # Convert to JAX-compatible model
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets the environment to an initial state."""
        rng, rng1 = jax.random.split(rng)

        data = mjx_env.init(
            self.mjx_model,
            qpos=qpos,
            qvel=qvel,
        )

    def _reset_model_pos(self) -> jax.Array:
        """Resets the model to an initial state."""
        qpos = jp.zeros(self.mjx_model.nq)
        return qpos        
    
def main():
    env = EnvWalt2D()
    env.reset(rng=jax.random.PRNGKey(0))

if __name__ == "__main__":
    main()