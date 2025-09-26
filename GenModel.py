import mujoco
import numpy as np
import os


# Class for generating a 2D WaLTER model programatically
class GenModel:
    def __init__(self):
        self.spec = mujoco.MjSpec()
        self.default_color = np.array([177/255, 166/255, 136/255, 1])

        model_params = {
            'segment_thickness': 0.05,
            'torso_len': 0.5,
            'thigh_len': 0.2,
            'shin_len': 0.3,
            'wheel_radius': 0.1,
            'segment_density': 10,  # kg/m^3
        }

        motor_params = {
            'hip_kp': 100,
            'hip_kd': 5,
            'hip_gear': 150,
            'hip_armature': 0.1,
            'knee_kp': 100,
            'knee_kd': 5,
            'knee_gear': 150,
            'knee_armature': 0.1,
            'wheel_kp': 50,
            'wheel_kd': 5,
            'wheel_gear': 70,
            'wheel_armature': 0.1,
        }

        self.model_params = model_params
        self.motor_params = motor_params

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

        # Create the torso body 
        torso_body = self.spec.worldbody.add_body(
            name = 'torso',
            pos = [0, 0, 1],
            quat = [1, 0, 0, 0],
        )
        torso_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [model_params['segment_thickness'], model_params['torso_len']/2, 0],
            quat = [1,0,1,0],
            mass = model_params['segment_density']*model_params['torso_len'],
            contype = body_contype,
            conaffinity = body_conaffinity,
            rgba = self.default_color,
        )
        # Create planar joints for 2D movement
        torso_body.add_joint(
            type = mujoco.mjtJoint.mjJNT_SLIDE,
            name = 'z_slide'
        )
        torso_body.add_joint(
            type = mujoco.mjtJoint.mjJNT_SLIDE,
            axis = [1, 0, 0],
            name = 'x_slide'
        )
        torso_body.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'y_rot'
        )

        ''' Assembling the front leg '''
        # Create front thigh
        front_thigh = torso_body.add_body(
            name = 'front_thigh',
            pos = [model_params['torso_len']/2 + model_params['thigh_len']/2, 0, 0],
            quat = [1, 0, 1, 0],
        )
        front_thigh.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [model_params['segment_thickness'], model_params['thigh_len']/2, 0],
            contype = thigh_contype,
            conaffinity = thigh_conaffinity,
        )
        front_thigh.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'front_hip',
            pos = [0, 0, -model_params['thigh_len']/2],
            armature = motor_params['hip_armature'],
        )
        # Create front shin
        front_shin = front_thigh.add_body(
            name = 'front_shin',
            pos = [0, 0, model_params['thigh_len']/2],
            quat = [1, 0, 1, 0],
        )
        front_shin.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [model_params['segment_thickness'], model_params['shin_len']/2, 0],
            contype = shin_contype,
            conaffinity = shin_conaffinity,
        )
        front_shin.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'front_knee',
            pos = [0, 0, 0],
            armature = motor_params['knee_armature'],
        )
        # Adding front wheel #1
        front_wheel1 = front_shin.add_body(
            name = 'front_wheel1',
            pos = [0, 0, model_params['shin_len']/2],
            quat = [1, 0, 0, 0],
        )
        front_wheel1.add_geom(
            type = mujoco.mjtGeom.mjGEOM_SPHERE,
            size = [model_params['wheel_radius'], 0, 0],
            contype = wheel_contype,
            conaffinity = wheel_conaffinity,
        )
        front_wheel1.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'front_wheel1',
            pos = [0, 0, 0],
            armature = motor_params['wheel_armature'],
        )
        # Adding front wheel #2
        front_wheel2 = front_shin.add_body(
            name = 'front_wheel2',
            pos = [0, 0, -model_params['shin_len']/2],
            quat = [1, 0, 0, 0],
        )
        front_wheel2.add_geom(
            type = mujoco.mjtGeom.mjGEOM_SPHERE,
            size = [model_params['wheel_radius'], 0, 0],
            contype = wheel_contype,
            conaffinity = wheel_conaffinity,
        )
        front_wheel2.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'front_wheel2',
            pos = [0, 0, 0],
            armature = motor_params['wheel_armature'],
        )

        ''' Assembling the rear leg '''
        # Create back thigh
        rear_thigh = torso_body.add_body(
            name = 'rear_thigh',
            pos = [-model_params['torso_len']/2 - model_params['thigh_len']/2, 0, 0],
            quat = [1, 0, 1, 0],
        )
        rear_thigh.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [model_params['segment_thickness'], model_params['thigh_len']/2, 0],
            contype = thigh_contype,
            conaffinity = thigh_conaffinity,
        )
        rear_thigh.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'back_hip',
            pos = [0, 0, model_params['thigh_len']/2],
            armature = motor_params['hip_armature'],
        )
        # Create rear shin
        rear_shin = rear_thigh.add_body(
            name = 'rear_shin',
            pos = [0, 0, -model_params['thigh_len']/2],
            quat = [1, 0, 1, 0],
        )
        rear_shin.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [model_params['segment_thickness'], model_params['shin_len']/2, 0],
            contype = shin_contype,
            conaffinity = shin_conaffinity,
        )
        rear_shin.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'back_knee',
            pos = [0, 0, 0],
            armature = motor_params['knee_armature'],
        )
        # Adding rear wheel #1
        rear_wheel1 = rear_shin.add_body(
            name = 'rear_wheel1',
            pos = [0, 0, -model_params['shin_len']/2],
            quat = [1, 0, 0, 0],
        )
        rear_wheel1.add_geom(
            type = mujoco.mjtGeom.mjGEOM_SPHERE,
            size = [model_params['wheel_radius'], 0, 0],
            contype = wheel_contype,
            conaffinity = wheel_conaffinity,
        )
        rear_wheel1.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'rear_wheel1',
            pos = [0, 0, 0],    
            armature = motor_params['wheel_armature'],
        )
        # Adding rear wheel #2
        rear_wheel2 = rear_shin.add_body(
            name = 'rear_wheel2',
            pos = [0, 0, model_params['shin_len']/2],
            quat = [1, 0, 0, 0],
        )
        rear_wheel2.add_geom(
            type = mujoco.mjtGeom.mjGEOM_SPHERE,
            size = [model_params['wheel_radius'], 0, 0],
            contype = wheel_contype,
            conaffinity = wheel_conaffinity,
        )
        rear_wheel2.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'rear_wheel2',
            pos = [0, 0, 0],
            armature = motor_params['wheel_armature'],
        )

        # Add actuators for the joints
        front_hip_act = self.add_position_actuator(
            'front_hip',
            motor_params['hip_kp'],
            motor_params['hip_kd'],
        )
        front_knee_act = self.add_velocity_actuator(
            'front_knee',
            motor_params['knee_kp'],
        )
        front_wheel1_act = self.add_velocity_actuator(
            'front_wheel1',
            motor_params['wheel_kp'],
        )
        front_wheel2_act = self.add_velocity_actuator(
            'front_wheel2',
            motor_params['wheel_kp'],
        )
        back_hip_act = self.add_position_actuator(
            'back_hip',
            motor_params['hip_kp'],
            motor_params['hip_kd'],
        )
        back_knee_act = self.add_velocity_actuator(
            'back_knee',
            motor_params['knee_kp'],
        )
        back_wheel1_act = self.add_velocity_actuator(
            'rear_wheel1',
            motor_params['wheel_kp'],
        )
        back_wheel2_act = self.add_velocity_actuator(
            'rear_wheel2',
            motor_params['wheel_kp'],
        )


    def add_position_actuator(self, joint_name, kp, kd):
        act = self.spec.add_actuator(
            name = joint_name + '_act',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            target = joint_name,
        )
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE

        act.gainprm[0] = kp
        act.biasprm[0:3] = [0.0, -kp, -kd]
        return act
        
    def add_velocity_actuator(self, joint_name, kv):
        act = self.spec.add_actuator(
            name = joint_name + '_act',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            target = joint_name,
        )
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE

        act.gainprm[0] = kv
        act.biasprm[0:3] = [0.0, 0.0, -kv]
        return act
    
    def add_scene(self):
        # Create ground plane texture/material
        ground = self.spec.add_texture(type = mujoco.mjtTexture.mjTEXTURE_2D,
                              name="ground_texture",
                              builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, 
                              width=200, 
                              height=200, 
                              rgb1=[0.5, 0.8, 0.9], 
                              rgb2=[0.5, 0.9, 0.8],
                              markrgb=[0.8, 0.8, 0.8])
        
        self.spec.add_material(name="groundplane",
                              texrepeat=[2, 2],
                              reflectance=0., 
                              ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'ground_texture'
        
        self.spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            material="groundplane",
        )


        # Create skybox so background isn't just black
        self.spec.add_texture(type = mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                              builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                                width = 300,
                                height = 300,
                                name="skybox")
        # Add an array of lights to the scene:
        for i in range(5):
            for j in range(5):
                self.spec.worldbody.add_light(
                    pos=[2*i, 2*j, 15],
                    dir=[0, 0, -1],
                    diffuse=[0.1, 0.1, 0.1],
                    specular=[0., 0., 0.],
                )
    

    def compile(self):
        self.model = mujoco.MjModel(self.spec)
        self.data = mujoco.MjData(self.model)
        return self.model, self.data
    
    def compile_to_xml(self, filename):
        self.spec.compile()
        xml_path = os.path.join(os.path.dirname(__file__), '2DWalt.xml')
        with open(xml_path, 'w') as f:
            f.write(self.spec.to_xml())
        print(f"Model XML saved to {xml_path}")

def main():
    walt = GenModel()
    walt.add_scene()
    walt.compile_to_xml('2DWalt.xml')


if __name__ == "__main__":
    main()