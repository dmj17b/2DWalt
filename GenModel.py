import mujoco
import numpy as np


# Class for generating a 2D WaLTER model programatically
class GenModel:
    def __init__(self):
        self.spec = mujoco.MjSpec()
        self.default_color = np.array([177/255, 166/255, 136/255, 1])

        model_params = {
            'segment_thickness': 0.05,
            'torso_len': 0.3,
            'thigh_len': 0.2,
            'shin_len': 0.2,
            'wheel_radius': 0.1,
            'segment_density': 10,  # kg/m^3
        }

        motor_params = {
            'hip_kp': 100,
            'hip_gear': 150,
            'hip_armature': 0.01,
            'knee_kp': 100,
            'knee_gear': 150,
            'knee_armature': 0.01,
            'wheel_kp': 50,
            'wheel_gear': 70,
            'wheel_armature': 0.01,
        }

        body_contype = 2
        body_conaffinity = 1
        thigh_contype = 2
        thigh_conaffinity = 1
        shin_contype = 2
        shin_conaffinity = 1
        wheel_contype = 4
        wheel_conaffinity = 5
        world_contype = 1
        world_conaffinity = 1

        torso_body = self.spec.add_body(
            name = 'torso',
            pos = [0, 0, 1],
            quat = [1, 0, 0, 0],
        )
        torso_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [model_params['segment_thickness'], model_params['torso_len']/2],
            quat = [1,0,1,0],
            mass = model_params['segment_density']*model_params['torso_len'],
            contype = body_contype,
            conaffinity = body_conaffinity,
            rgba = self.default_color,
        )

    def compile(self):
        self.model = mujoco.MjModel(self.spec)
        self.data = mujoco.MjData(self.model)
        return self.model, self.data
    def compile_to_xml(self, filename):
        self.spec.compile()
        xml_path = os.path.join(os.path.dirname(__file__), '2DWalt.xml')

def main():
    walt = GenModel()
    walt.compile_to_xml('2DWalt.xml')


if __name__ == "__main__":
    main()