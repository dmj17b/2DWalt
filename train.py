import EnvWalt2D
from datetime import datetime
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.io import model
from mujoco_playground import wrapper
import functools
import jax
from mujoco_playground.config import dm_control_suite_params
import wandb

env = EnvWalt2D.EnvWalt2D()  # Create an instance of the EnvWalt2D environment
env_cfg = env.config  # Retrieve the environment configuration
ppo_params = {
    'action_repeat': 5,
    'batch_size': 1024,
    'discounting': 0.995,
    'entropy_cost': 0.001,
    'episode_length': 1000,
    'learning_rate': 1e-4,
    'num_envs': 4096,
    'num_evals': 20,  
    'num_minibatches': 32,
    'num_updates_per_batch': 8,
    'num_timesteps': 100_000_000,  
    'normalize_observations': True,
    'reward_scaling': 1.0,
    'unroll_length': 50,
    }

#---------- WandB logging setup ------------#
wandb.login()
project = "2DWalt_PPO"
wandb_config = dict(ppo_params)
run = wandb.init(project=project, config=wandb_config)

def _to_float(value):
    """Safely converts JAX/NumPy scalars to plain Python floats for logging."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

# Progress callback
def progress(num_steps, metrics):
    print(f"Steps: {num_steps}")
    print(f"Body pitch reward: {metrics['eval/episode_reward/body_pitch']:.4f}")
    print(f"Velocity tracking reward: {metrics['eval/episode_reward/vel_tracking']:.4f}")
    print(f"Torque Penalty: {metrics['eval/episode_reward/low_torques']:.6f}")
    print(f"Body z-velocity penalty: {metrics['eval/episode_reward/body_z_vel']:.4f}")
    print(f"Body pitch velocity penalty: {metrics['eval/episode_reward/body_pitch_vel']:.4f}")
    print(f"Total reward: {metrics['eval/episode_reward']:.4f}\n")

    wandb_metrics = {k: _to_float(v) for k, v in metrics.items()}
    wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
    wandb_metrics["num_steps"] = int(num_steps)
    run.log(wandb_metrics, step=int(num_steps))
    

ppo_training_params = dict(ppo_params)

network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_training_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory,
    )

train_fn = functools.partial(
    ppo.train,
    **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress
)

print("Starting PPO training...")
print(f"Total timesteps: {ppo_params['num_timesteps']}")
print(f"Num environments: {ppo_params['num_envs']}")
print(f"Steps per iteration: {ppo_params['num_envs'] * ppo_params['unroll_length']}")
print(f"Expected iterations: ~{ppo_params['num_timesteps'] // (ppo_params['num_envs'] * ppo_params['unroll_length'])}")
print("-" * 60)

make_inference_fn, params, metrics = train_fn(
    environment = env,
    wrap_env_fn = wrapper.wrap_for_brax_training
)

run.finish()  # Finish the WandB run after training is complete
    
print(f"\nFinal metrics: {metrics}")

# Save the trained model parameters using Brax's model saving utility
model_path = "walter_ppo"
model.save_params(model_path, params)