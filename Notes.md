# TODO:
- Add timed command changes and zero-velocity commands
- Add reward configs to wandb
- Randomized obstacle "generation"
    - Start with N obstacles, all "underground" then randomize their heights with each test
- Refactor config inclusions for better reusability
    - Add environment, reward, and command configs to WandB so I remember what has changed
- (Eventually) add distance sensors at front and rear of body
- Create training curriculum that slowly introduces obstacles
- (Eventually) Make environment work with MJWarp
    - Supposedly warp has better contact modeling than mjx and allows for different geoms
- Add in standstill penalties when velocity setpoint is zero

# Ideas:
- Warm start policy with data from joystick control?
- Penalize hip/knee torque more than wheels (weighted torque )
- Add joint torques to observation (to help policy 'feel' when the robot hits an obstacle)

# Need to understand better:
- Cement understanding of policy rollouts, network updates, etc.

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