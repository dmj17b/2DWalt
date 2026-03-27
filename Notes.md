# TODO:
- Wrap knee angles between 0 and 2*pi (relative to hip?)
- sin/cos of body angle in observation
- Implement exponential kernel
- Separate config file for reward/ppo params
- Add velocity tracking term that changes with each reset

# Questions for Jacob:
- Mentioned that actions were sampled from a mean and std deviation
    - Where does this happen? And is that not just for training? 
    - I would think that policy execution on hardware (or for evaluation) is deterministic