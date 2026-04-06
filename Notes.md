# TODO:
- Add timed command changes and zero-velocity commands
- Add reward configs to wandb
- Randomized obstacle "generation"
    - Start with N obstacles, all "underground" then randomize their heights with each test
- Refactor config inclusions for better reusability
- Switch velocity tracking to BODY frame instead of world frame
- (Eventually) add distance sensors at front and rear of body
- Add friction and reflected inertia to joints

# Ideas:
- Warm start policy with data from joystick control?
- Penalize hip/knee torque more than wheels (weighted torque )

# Questions for Jacob:
