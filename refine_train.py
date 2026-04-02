import jax
import jax.numpy as jp
import mujoco as mj
import mujoco.viewer
import time
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from brax.io import model
from ml_collections import config_dict
import functools
from mujoco_playground import wrapper
import inspect
from pathlib import Path
import wandb

import EnvWalt2D

    
def main():

    resume_path = "walter_ppo"  # Path to the saved PPO model parameters to resume training from
    resume_params = model.load_params(resume_path)

    env = EnvWalt2D.EnvWalt2D()  # Create an instance of the EnvWalt2D environment
    env_cfg = env.config  # Retrieve the environment configuration
    ppo_params = {
        'action_repeat': 1,
        'batch_size': 2048,
        'discounting': 0.995,
        'entropy_cost': 0.0001,
        'episode_length': 1000,
        'learning_rate': 5e-5,
        'num_envs': 4096,
        'num_evals': 10,  
        'num_minibatches': 64,
        'num_updates_per_batch': 8,
        'num_timesteps': 10_000_000,  
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
        print(f"Height penalty: {metrics['eval/episode_reward/height_penalty']:.4f}")
        print(f"Action smoothing penalty: {metrics['eval/episode_reward/action_smoothing']:.6f}")
        print(f"Total reward: {metrics['eval/episode_reward']:.4f}\n")

        wandb_metrics = {k: _to_float(v) for k, v in metrics.items()}
        wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
        wandb_metrics["num_steps"] = int(num_steps)
        run.log(wandb_metrics, step=int(num_steps))
        

    
    ppo_training_params = ppo_params
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]  # Remove network factory from training params since it is not a valid argument for the PPO class
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, 
            **ppo_params.network_factory
        )
    
    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory = network_factory,
        progress_fn=progress,
    )
    train_kwargs = dict(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    make_inference_fn, params, metrics = train_fn(
        **train_kwargs,
        restore_params=resume_params
    )

    # Save the trained policy parameters and metrics:
    model_path = "walter_ppo_retrain"
    model.save_params(model_path, params)



if __name__ == "__main__":
    main()  
