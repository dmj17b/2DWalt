import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to pathimport mujoco
import mujoco.viewer
import jax
import jax.numpy as jp
import time
import modeling.GenModel as GenModel
import numpy as np

model_spec = GenModel.GenModel()  # Create an instance of the model generator
model_spec.add_scene()  # Add the scene to the model
# model_spec.add_hfield()  # Add a heightfield to the model for testing
model_spec.add_groundplane()  # Add a ground plane to the model for testing
model_spec.add_box_obstacles()

mj_model = model_spec.spec.compile()
mj_data = mujoco.MjData(mj_model)

def randomize_boxes(mj_model, 
                    mj_data, 
                    n_boxes = 10, 
                    z_range = (-0.2, 0.1),
                    x_range = (-15, 15)):
    # Randomize the height and position of each box obstacle
    for i in range(n_boxes):
        box_body_id = mj_model.body(f"box_{i}").id
        box_mocap_id = mj_model.body_mocapid[box_body_id]
        # Randomize the height of the box within the specified range
        box_height = np.random.uniform(z_range[0], z_range[1])
        # Randomize the x position of the box within the specified range
        x = np.linspace(x_range[0], x_range[1], n_boxes)[i]
        mj_data.mocap_pos[box_mocap_id] = [x, 0.0, box_height]



# Launch standard MuJoCo viewer
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        # Keep track of step time
        start_time = time.time()

        viewer.sync()  # Sync the viewer to update the visualization


        mujoco.mj_step(mj_model, mj_data)  # Step the simulation forward

        # Randomize box positions and heights every 100 steps for testing purposes
        if mj_data.time > 0 and int(mj_data.time / 0.01) % 100 == 0:
            randomize_boxes(mj_model, mj_data)


        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = (mj_model.opt.timestep - (time.time() - start_time))
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
