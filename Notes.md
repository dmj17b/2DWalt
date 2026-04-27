# Algorithm Tuning TODOs:
- Figure out how to define/execute curriculums
- Add in standstill penalties for zero velocity commands (learn how to stay still)
- (Maybe) Penalize deviation from 'standard' hip/knee angles?
- Add difficulty to box environment (vary max height)
- Increase spacing between box obstacles so the policy still learns flat ground

# Code Cleanup TODOs:
- Add reward configs to wandb
- Refactor config inclusions for better reusability
    - Add environment, reward, and command configs to WandB so I remember what has changed
- (Eventually) add distance sensors at front and rear of body
- (Eventually) Make environment work with MJWarp
    - Supposedly warp has better contact modeling than mjx and allows for different geoms

# Ideas:
- Warm start policy with data from joystick control?
- Penalize hip/knee torque more than wheels (weighted torque penalties based on torque constant)

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

# Ideas for Experimentation:
- Create a COT or total energy function for grounding simulations to reality