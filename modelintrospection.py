import mujoco # Import mujoco for model creation and manipulation.
import jax  # Import JAX for numerical computing.
import jax.numpy as jp  # Import JAX's numpy as jp for array operations.
from typing import Optional, Dict, Union  # Import typing utilities for type hints.
import EnvWalt2D  # Import the EnvWalt2D environment modulle
import GenModel  # Import the GenModel module for model generation.
import mujoco.viewer  # Import mujoco viewer for visualization.
import time  # Import time module for timing operations.
import numpy as np  # Import numpy for numerical operations.


model_spec = GenModel.GenModel()  # Create an instance of the model generator
model_spec.add_scene()  # Add the scene to the model
m = model_spec.compile_mj_model()  # Compile the model and retrieve it
m.opt.enableflags
d = mujoco.MjData(m)  # Create a data object for the model

with mujoco.viewer.launch_passive(m,d) as viewer:
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # Step the simulation forward
        mujoco.mj_step(m, d)

        # Test control inputs:
        d.ctrl[0] = 0.5
        d.ctrl[1] = 1.5
        d.ctrl[4] = -0.5
        d.ctrl[5] = 1.5

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
        

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)