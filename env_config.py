import jax
import jax.numpy as jp
import flax.struct

@flax.struct.dataclass
class RewardConfig:
    fwd_vel_weight = 1.0
    body_pitch_weight = -1.0
    low_torques_weight = -0.0005
    alive = 0.0
    termination = -100.0