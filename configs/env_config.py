import jax
import jax.numpy as jp
import flax.struct
import os
import sys
from ml_collections import config_dict  # Import config_dict for configuration management.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path

@flax.struct.dataclass
class RewardConfig:
    vel_tracking: float = 100.0
    body_pitch: float = 8.0
    body_pitch_vel: float = 10.0
    low_torques: float = 0.001
    body_z_vel: float = 10.0
    height_penalty: float = 10.0
    action_smoothing: float = 10.0
    terminal_pitch: float = 100.0
    joint_vel: float = 10.0

@flax.struct.dataclass
class CommandConfig:
    max_vel: float = 1.5
    min_cmd_duration: float = 0.75
    max_cmd_duration: float = 2.0
    zero_cmd_prob: float = 0.1

def SimConfig() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt = 0.02,
        sim_dt = 0.002,
        episode_length = 2000,
        action_repeat = 1,
        impl = 'warp',
        naconmax = 10*4096,
    )

@flax.struct.dataclass
class TerrainConfig:
    terrain_type: str = "flat" # Options: "flat", "heightfield", "boxes", "stairs"
    difficulty_start: float = 0.0

@flax.struct.dataclass
class TrainConfig:
    action_repeat: int = 1
    batch_size: int = 1024
    discounting: float = 0.995
    entropy_cost: float = 0.01
    episode_length: int = 1000
    learning_rate: float = 1e-4
    num_envs: int = 4096
    num_evals: int = 100
    num_minibatches: int = 32
    num_updates_per_batch: int = 16
    num_timesteps: int = 1_000_000
    normalize_observations: bool = True
    reward_scaling: float = 1.0
    unroll_length: int = 30

