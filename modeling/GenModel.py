import mujoco
import numpy as np
import os
import sys
import jax
import jax.numpy as jp
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path

# Class for generating a 2D WaLTER model programatically
class GenModel:
    def __init__(self):
        self.spec = mujoco.MjSpec()
        self.default_color = np.array([177/255, 166/255, 136/255, 1])

        model_params = {
            'segment_thickness': 0.05,
            'torso_len': 0.6,
            'thigh_len': 0.2,
            'shin_len': 0.3,
            'wheel_radius': 0.12,
            'segment_density': 10,  # kg/m^3
        }

        motor_params = {
            'hip_kp': 100,
            'hip_kd': 25,
            'hip_gear': 150,
            'hip_armature': 0.4,
            'hip_frictionloss': 0.1,
            'knee_kp': 100,
            'knee_kd': 25,
            'knee_gear': 150,
            'knee_armature': 0.4,
            'knee_frictionloss': 0.1,
            'wheel_kp': 100,
            'wheel_kd': 25,
            'wheel_gear': 70,
            'wheel_armature': 0.2,
            'wheel_frictionloss': 0.01,
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
            pos = [0, 0, 0.3],
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
            axis = [1, 0, 0],
            name = 'x_slide'
        )
        torso_body.add_joint(
            type = mujoco.mjtJoint.mjJNT_SLIDE,
            name = 'z_slide'
        )

        torso_body.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'y_rot'
        )

        torso_body.add_site(
            name = 'torso_center',
            pos = [0, 0, 0]
        )

        ''' Assembling the front leg '''
        # Create front thigh
        front_thigh = torso_body.add_body(
            name = 'front_thigh',
            pos = [model_params['torso_len']/2 , 0, -model_params['thigh_len']/2],
            quat = [0, 0, 1, 0],
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
            frictionloss = motor_params['hip_frictionloss'],
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
            frictionloss = motor_params['knee_frictionloss'],
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
            frictionloss = motor_params['wheel_frictionloss'],
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
            frictionloss = motor_params['wheel_frictionloss'],
        )

        ''' Assembling the rear leg '''
        # Create rear thigh
        rear_thigh = torso_body.add_body(
            name = 'rear_thigh',
            pos = [-model_params['torso_len']/2, 0, -model_params['thigh_len']/2],
            quat = [1, 0, 0, 0],
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
            name = 'rear_hip',
            pos = [0, 0, model_params['thigh_len']/2],
            armature = motor_params['hip_armature'],
            frictionloss = motor_params['hip_frictionloss'],
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
            name = 'rear_knee',
            pos = [0, 0, 0],
            armature = motor_params['knee_armature'],
            frictionloss = motor_params['knee_frictionloss'],
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
            frictionloss = motor_params['wheel_frictionloss'],
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
            frictionloss = motor_params['wheel_frictionloss'],
        )

        # Add actuators for the joints
        front_hip_act = self.add_position_actuator(
            'front_hip',
            motor_params['hip_kp'],
            motor_params['hip_kd'],
        )
        front_knee_act = self.add_position_actuator(
            'front_knee',
            motor_params['knee_kp'],
            motor_params['knee_kd'],
        )
        front_wheel1_act = self.add_velocity_actuator(
            'front_wheel1',
            motor_params['wheel_kp'],
        )
        front_wheel2_act = self.add_velocity_actuator(
            'front_wheel2',
            motor_params['wheel_kp'],
        )
        rear_hip_act = self.add_position_actuator(
            'rear_hip',
            motor_params['hip_kp'],
            motor_params['hip_kd'],
        )
        rear_knee_act = self.add_position_actuator(
            'rear_knee',
            motor_params['knee_kp'],
            motor_params['knee_kd'],
        )
        rear_wheel1_act = self.add_velocity_actuator(
            'rear_wheel1',
            motor_params['wheel_kp'],
        )
        rear_wheel2_act = self.add_velocity_actuator(
            'rear_wheel2',
            motor_params['wheel_kp'],
        )

        body_vel_sensor = self.add_velocity_sensor('torso_center')

     

    # Helper function to add position actuator 
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
        
    # Helper function to add velocity actuator
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
    
    def add_velocity_sensor(self, site_name):
        sensor = self.spec.add_sensor(
            name = 'body_lin_vel',
            type = mujoco.mjtSensor.mjSENS_VELOCIMETER,
            objname = site_name,
            objtype = mujoco.mjtObj.mjOBJ_SITE,
        )
        return sensor
    
    def add_scene(self):
        # Create skybox so background isn't just black
        self.spec.add_texture(type = mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                              builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                                width = 300,
                                height = 300,
                                name="skybox",
                                rgb2 = [0.4, 0.7, 0.9],)
        # Ground plane texture/material
        self.spec.add_material(name="groundplane_material",
                        texrepeat=[2, 2],
                        reflectance=0., 
                        ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'ground_texture'
        # Create ground plane texture/material
        ground = self.spec.add_texture(type = mujoco.mjtTexture.mjTEXTURE_2D,
                              name="ground_texture",
                              builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, 
                              width=200, 
                              height=200, 
                              rgb1=[0.5, 0.8, 0.9], 
                              rgb2=[0.5, 0.9, 0.8],
                              markrgb=[0.8, 0.8, 0.8])
        # Add an array of lights to the scene:
        for i in range(15):
            for j in range(15):
                self.spec.worldbody.add_light(
                    pos=[3*i, 3*j, 50],
                    dir=[0, 0, -1],
                    diffuse=[0.1, 0.1, 0.1],
                    specular=[0.1, 0.1, 0.1],
                    intensity=1.0,
                )
                
    def add_groundplane(self):
        self.spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            material="groundplane_material",
        )

    def add_hfield(self, 
                   height: float = 0.5,
                   sigma: float = 0.6,
                   rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
                   ):
        # Add a heightfield to the environment for testing using perlin noise:
        nrow, ncol = 128, 128
        size = [30.0, 30.0, height, 0.5]  # [x_span, y_span, z_height, base_offset]
        # Add some random noise to the height field
        heightfield_data = np.zeros((nrow, ncol))
        rng = np.random.default_rng(seed=42)
        heightfield_data = heightfield_data + rng.uniform(0, height, size=(nrow, ncol)) 
        heightfield_data = gaussian_filter(heightfield_data, sigma=sigma)  # Smooth the heightfield with a Gaussian filter
        hfdata_flat = heightfield_data.flatten()
        self.spec.add_hfield(
            name='terrain',
            nrow=nrow,
            ncol=ncol,
            size=size,
            userdata=hfdata_flat,
        )
        ground = self.spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            name="hfield_texture",
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=200,
            height=200,
            rgb1=[0.5, 0.8, 0.95],
            rgb2=[0.5, 0.95, 0.8],
            markrgb=[0.8, 0.8, 0.8]
        )

        # Add material for heightfield
        self.spec.add_material(
            name="hfield_material",
            texrepeat=[5, 5],
            reflectance=0.0,
        ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'hfield_texture'
        terrain_body = self.spec.worldbody.add_body(
            name='terrain_body', 
            pos=[0, 0, 0],
            mocap = True,)
        terrain_body.add_geom(
            name='groundplane',
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname='terrain',
            pos=[0, 0, -0.75],
            material='hfield_material',
        ) 

    def add_stadium_hfield(self,
                           step_height: float = 0.3,
                           step_interval: int = 6,
                           max_height: float = 5.0,
                           sigma: float = 0.6,
                           rng: jax.random.PRNGKey = jax.random.PRNGKey(0,),
                           ):
        # Create "Stadium" style stepped heights:
        size = [30.0, 30.0, max_height, 0.5]  # [x_span, y_span, z_height, base_offset]
        nrow, ncol = 256, 256
        hfield_data = np.zeros((nrow, ncol))
        curr_height = max_height
        direction = -1  # Start by stepping downwards
        for i in range(nrow):
            if i % step_interval == 0:
                curr_height += step_height*direction
            # Reverse step direction at halfway point
            if i == nrow//2:
                direction *= -1
            hfield_data[:, i] = curr_height

        hfdata_flat = hfield_data.flatten()

        self.spec.add_hfield(
            name='terrain',
            nrow=nrow,
            ncol=ncol,
            size=size,
            userdata=hfdata_flat,
        )
        ground = self.spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            name="hfield_texture",
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=200,
            height=200,
            rgb1=[0.5, 0.8, 0.95],
            rgb2=[0.5, 0.95, 0.8],
            markrgb=[0.8, 0.8, 0.8]
        )

        # Add material for heightfield
        self.spec.add_material(
            name="hfield_material",
            texrepeat=[5, 5],
            reflectance=0.0,
        ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'hfield_texture'
        terrain_body = self.spec.worldbody.add_body(
            name='terrain_body', 
            pos=[0, 0, 0],
            mocap = True,)
        terrain_body.add_geom(
            name='groundplane',
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname='terrain',
            pos=[0, 0, -0.75],
            material='hfield_material',
        ) 

    def add_stepped_hfield(self, 
                    height: float = 8.0,
                    noise_height: float = 0.25,
                    step_height: float = 0.3,
                    step_interval: int = 8,
                    sigma: float = 0.6,
                    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
                   ):

        
        # Add a heightfield to the environment for testing using perlin noise:
        nrow, ncol = 128, 128
        size = [30.0, 30.0, height, 0.5]  # [x_span, y_span, z_height, base_offset]
        # Add some random steps to the height field
        heightfield_data = np.zeros((nrow, ncol))
        curr_height = 0.0
        for i in range(nrow):
            if i % step_interval == 0:  # Add a step every 8 rows
                curr_height += step_height  # Increase the height of the step
            heightfield_data[:, i] = curr_height
            heightfield_data[i, :] = curr_height  # Add the step height to the heightfield data

        rng = np.random.default_rng(seed=42)
        heightfield_data = heightfield_data + rng.uniform(0, noise_height, size=(nrow, ncol)) 
        heightfield_data = gaussian_filter(heightfield_data, sigma=sigma)  # Smooth the heightfield with a Gaussian filter
        hfdata_flat = heightfield_data.flatten()

        self.spec.add_hfield(
            name='terrain',
            nrow=nrow,
            ncol=ncol,
            size=size,
            userdata=hfdata_flat,
        )
        ground = self.spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            name="hfield_texture",
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=200,
            height=200,
            rgb1=[0.5, 0.8, 0.95],
            rgb2=[0.5, 0.95, 0.8],
            markrgb=[0.8, 0.8, 0.8]
        )

        # Add material for heightfield
        self.spec.add_material(
            name="hfield_material",
            texrepeat=[5, 5],
            reflectance=0.0,
        ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'hfield_texture'
        terrain_body = self.spec.worldbody.add_body(
            name='terrain_body', 
            pos=[0, 0, 0],
            mocap = True,)
        terrain_body.add_geom(
            name='groundplane',
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname='terrain',
            pos=[0, 0, -0.75],
            material='hfield_material',
        ) 

    def add_box_heightfield(self,
                            min_height: float = 0.1,
                            max_height: float = 0.4,
                            rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
                            sigma: float = 0.6,
                            ):
        
        # Add a heightfield to the environment for testing using perlin noise:
        nrow, ncol = 2, 1024
        size = [30.0, 10.0, max_height, 0.1]  # [x_span, y_span, z_height, base_offset]
        # Add some random noise to the height field
        heightfield_data = np.zeros((nrow, ncol))
        
        for i in range(0, ncol, 32):
            # Random integer for random width:
            width_rng, height_rng, rng = jax.random.split(rng, 3)
            width = jax.random.randint(width_rng, shape=(2,), minval=2, maxval=12)
            height = jax.random.uniform(height_rng, minval=min_height, maxval=1.0)
            heightfield_data[:,i-width[0]:i+width[1]] = height  # Create boxy steps every 8 rows

        hfdata_flat = heightfield_data.flatten()
        self.spec.add_hfield(
            name='terrain',
            nrow=nrow,
            ncol=ncol,
            size=size,
            userdata=hfdata_flat,
        )
        ground = self.spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            name="hfield_texture",
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=200,
            height=200,
            rgb1=[0.5, 0.8, 0.95],
            rgb2=[0.5, 0.95, 0.8],
            markrgb=[0.8, 0.8, 0.8]
        )

        # Add material for heightfield
        self.spec.add_material(
            name="hfield_material",
            texrepeat=[5, 5],
            reflectance=0.0,
        ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'hfield_texture'
        terrain_body = self.spec.worldbody.add_body(
            name='terrain_body', 
            pos=[0, 0, 0],
            mocap = True,)
        terrain_body.add_geom(
            name='groundplane',
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname='terrain',
            pos=[0, 0, -0.75],
            material='hfield_material',
        ) 
    
    def add_box_obstacles(self, 
                          n_boxes: int = 10, 
                          x_range: float = 20.0,
                          width_range: jp.ndarray = jp.array([0.15, 1.0]),
                          max_height: float = 0.25,
                          rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
                          ):
        x_pos = jp.linspace(-x_range, x_range, n_boxes)
        for i in range(n_boxes):
            rng, pos_rng, size_rng = jax.random.split(rng, 3)
            box = self.spec.worldbody.add_body(
                name=f'box_{i}',
                pos=[x_pos[i], 0.0, max_height/2],
                mocap = True,
            )
            box.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[jax.random.uniform(size_rng, minval=width_range[0], maxval=width_range[1]), 5, max_height],
                rgba=[0.8, 0.8, 0.8, 1],
                contype=1,
                conaffinity=0,
            )


    def compile_mj_model(self):
        self.model = self.spec.compile()
        return self.model
    
    def compile_to_xml(self, filename):
        self.spec.compile()
        xml_path = os.path.join(os.path.dirname(__file__), '2DWalt.xml')
        with open(xml_path, 'w') as f:
            f.write(self.spec.to_xml())
        print(f"Model XML saved to {xml_path}")

def main():
    walt = GenModel()
    walt.add_scene()
    walt.add_hfield()
    walt.compile_to_xml('2DWalt.xml')
    

if __name__ == "__main__":
    main()