# Priorities
- Visualization rollout methods
- Try out different wrappers for handling curriculum logic
- Organize config files!!!
- Figure out how backend rollouts work
- Normalize rewards to start at zero
- NN structure (512, 256, 128)
- Asymmetric observation space for policy vs value networks
- Domain Randomization


# Algorithm Tuning TODOs
- Figure out how to define/execute curriculums

# Code Cleanup TODOs:
- (Eventually) add distance sensors at front and rear of body


# Ideas:
- Warm start policy with data from joystick control?
- Penalize hip/knee torque more than wheels (weighted torque penalties based on torque constant)

# Need to understand better:
- Cement understanding of policy rollouts, network updates, etc.
- Environment wrappers and vectorization

# Organization:
- Model
    - GenModel.py
    - model_test.py (joystick control, standard mujoco implementation)
- Env
    - EnvWalt2D.py
- Training
- Testing
    - env_test.py   (test environment wrapper, rewards, reset, etc.)
    - js_env_test.py (env_test but with joystick control)
    - policy_test.py  (simple policy test following same reset protocol as training)
    - js_policy_test.py (continuous policy test with joystick mapped to velocity setpoint)
- Policies
    - walter_ppo...

# Ideas for Experimentation:
- Create a COT or total energy function for grounding simulations to reality

# Jacob Recommendations:

- Try velocities on knees - see what happens
- Asymmetric actor/critic observations 
- "encoder" framework - big to small (worry about this later)
- Try rangefinder (restrict max distance observation)
- Zero velocity penalties
- Body pitch penalty separate from task reward

