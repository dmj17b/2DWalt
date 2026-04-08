import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to pathimport mujoco
import mujoco.viewer
import jax
import jax.numpy as jp
import time
import modeling.GenModel as GenModel

model_spec = GenModel.GenModel()  # Create an instance of the model generator
model_spec.add_scene()  # Add the scene to the model
model_spec.add_hfield()  # Add a heightfield to the model for testing

mj_model = model_spec.spec.compile()
mj_data = mujoco.MjData(mj_model)
# Launch standard MuJoCo viewer
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        # Keep track of step time
        start_time = time.time()

        viewer.sync()  # Sync the viewer to update the visualization


        mujoco.mj_step(mj_model, mj_data)  # Step the simulation forward


        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = (mj_model.opt.timestep - (time.time() - start_time))
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
