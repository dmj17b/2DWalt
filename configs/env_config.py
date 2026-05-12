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
    success_bonus: float = 1000.0

class StairRewardConfig(RewardConfig):
    body_pitch: float = 12.0  # Increase the weight on body pitch to encourage the agent to maintain an upright posture while climbing stairs
    body_pitch_vel: float = 15.0  # Increase the weight on body pitch velocity to encourage smoother and more controlled movements during stair climbing
    low_torques: float = 0.001  # Increase the penalty for high torques to encourage more efficient and careful movements on stairs
    body_z_vel: float = 15.0  # Increase the weight on vertical velocity to encourage the agent to focus on upward movement when climbing stairs
    height_penalty: float = 20.0  # Increase the penalty for low height to encourage the agent to maintain a higher position while climbing stairs
    terminal_pitch: float = 150.0  # Increase the penalty for falling over to strongly discourage the agent from losing balance on stairs
    success_bonus: float = 1500.0  # Increase the success bonus to provide a stronger incentive for successfully climbing the stairs
    pos_reward: float = 10.0  # Add a reward for forward progress
    

@flax.struct.dataclass
class CommandConfig:
    max_vel: float = 1.5
    min_cmd_duration: float = 1.5
    max_cmd_duration: float = 3.0
    zero_cmd_prob: float = 0.1

class StairCommandConfig(CommandConfig):
    max_vel: float = 1.0  # Reduce max velocity for stair climbing tasks to encourage careful navigation
    min_cmd_duration: float = 29.0  # Increase minimum command duration to encourage sustained commands for stair climbing
    max_cmd_duration: float = 30.0  # Increase maximum command duration to allow for longer sustained commands during stair climbing

def SimConfig() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt = 0.02,
        sim_dt = 0.004,
        episode_length = 3000,
        action_repeat = 1,
        impl = 'warp',
        naconmax = 10*4096,
    )

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

