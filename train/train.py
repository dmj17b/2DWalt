import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
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

import environment.BoxEnv as BoxEnv
import environment.HFieldEnv as HFieldEnv
import environment.FlatEnv as FlatEnv

    
def main():

    resume_path = None  # Path to the saved PPO model parameters to resume training from
    save_path = "policies/walter_ppo_warp"  # Path to save the new PPO model parameters after training

    # env = FlatEnv.FlatEnv()  # Create an instance of the FlatEnv environment with a moderate difficulty level
    env = HFieldEnv.HFieldEnv(difficulty=0.1)  # Create an instance of the HFieldEnv environment with a moderate difficulty level
    # env = BoxEnv.BoxEnv(difficulty=0.75)  # Create an instance of the BoxEnv environment
    env_cfg = env.config  # Retrieve the environment configuration
    ppo_params = {
        'action_repeat': 1,
        'batch_size': 4096,  
        'discounting': 0.995,
        'entropy_cost': 0.01,
        'episode_length': env_cfg.episode_length,
        'learning_rate': 1e-4,
        'num_envs': 4096,
        'num_evals': 20,  
        'num_minibatches': 32,
        'num_updates_per_batch': 4,
        'num_timesteps': 100_000_000,  
        'normalize_observations': True,
        'reward_scaling': 1.0,
        'unroll_length': 32,
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
        if not hasattr(progress, "eval_counter"):
            progress.eval_counter = 0

        # FIX: Print the available keys on the first run so you know exactly what Brax named them
        if progress.eval_counter == 0:
            print("Available Metric Keys:", list(metrics.keys()))

        print(f"\nEvaluation #{progress.eval_counter}:")
        print(f"Steps: {num_steps}")
        print(f"Task reward: {metrics.get('eval/episode_reward/task', 0.0):.4f}")
        print(f"Velocity tracking reward: {metrics.get('eval/episode_reward/vel_tracking', 0.0):.4f}")
        print(f"Body pitch penalty: {metrics.get('eval/episode_reward/body_pitch', 0.0):.4f}")
        print(f"Body pitch velocity penalty: {metrics.get('eval/episode_reward/body_pitch_vel', 0.0):.4f}")
        print(f"Body z velocity penalty: {metrics.get('eval/episode_reward/body_z_vel', 0.0):.4f}")
        print(f"Torque Penalty: {metrics.get('eval/episode_reward/low_torques', 0.0):.6f}")
        print(f"Action smoothing penalty: {metrics.get('eval/episode_reward/action_smoothing', 0.0):.6f}")
        print(f"Total reward: {metrics.get('eval/episode_reward', 0.0):.4f}")

        wandb_metrics = {k: _to_float(v) for k, v in metrics.items()}
        wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
        wandb_metrics["num_steps"] = int(num_steps)
        run.log(wandb_metrics, step=int(num_steps))
        
        progress.eval_counter += 1
        

    
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
    # If a resume path is provided, load the parameters and pass them, otherwise start training from scratch
    if resume_path is not None and Path(resume_path).exists():
        print(f"Resuming PPO training from {resume_path}...")
        resume_params = model.load_params(resume_path)
        make_inference_fn, params, metrics = train_fn(
           **train_kwargs,
            restore_params=resume_params
        )
    else:
        print(f"Starting PPO training from scratch...")
        make_inference_fn, params, metrics = train_fn(
            **train_kwargs
        )

    run.finish()  # Finish the WandB run after training is complete
    print(f"\nFinal metrics: {metrics}")

    # Save the trained policy parameters and metrics:
    model.save_params(save_path, params)



if __name__ == "__main__":
    main()  
