import jax
import jax.numpy as jp
import flax.struct

@flax.struct.dataclass
class RewardConfig:
    vel_tracking = 100.0
    body_pitch = 50.0
    low_torques = -0.001
    body_height = 0.0
    alive = 0.0
    termination = -100.0

@flax.struct.dataclass
class EnvConfig:
    reward_config: RewardConfig = RewardConfig()
    action_repeat: int = 1
    impl: str = 'jax'
    action_scale: float = 0.5
    ctrl_dt = 0.01
    sim_dt = 0.002

@flax.struct.dataclass
class TrainConfig:
    action_repeat: int = 1,
    batch_size: int = 1024,
    discounting: float = 0.995,
    entropy_cost: float = 0.01,
    episode_length: int = 1000,
    learning_rate: float = 1e-4,
    num_envs: int = 4096,
    num_evals: int = 100,
    num_minibatches: int = 32,
    num_updates_per_batch: int = 16,
    num_timesteps: int = 1_000_000,
    normalize_observations: bool = True,
    reward_scaling: float = 1.0,
    unroll_length: int = 30,

