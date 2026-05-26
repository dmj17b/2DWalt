import os
import sys

from wandb.util import np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from environment import CurriculumWrapper

import jax
import jax.numpy as jp
import mujoco as mj
import mujoco.viewer
import time
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax.io import html
from ml_collections import config_dict
import functools
from mujoco_playground import wrapper
import inspect
from pathlib import Path
import wandb
import visualization.HTML_Renderer as HTML_Renderer
import environment.BoxEnv as BoxEnv
import environment.HFieldEnv as HFieldEnv
import environment.FlatEnv as FlatEnv
import environment.StairEnv as StairEnv

    
def main():

    resume_path = "policies/stairs"  # Path to the saved PPO model parameters to resume training from
    save_path = "policies/stairs2"  # Path to save the new PPO model parameters after training

    notes = "Got rid of xpos reward and zpos reward. Increased learning rate. Changed to start at challenge level 2 since that's where previous agent failed"

    # env = FlatEnv.FlatEnv()  # Create an instance of the FlatEnv environment with a moderate difficulty level
    # env = HFieldEnv.HFieldEnv(difficulty=0.25)  # Create an instance of the HFieldEnv environment with a moderate difficulty level
    # env = BoxEnv.BoxEnv(difficulty=0.9, spacing=48)  # Create an instance of the BoxEnv environment
    env = StairEnv.StairEnv(challenge_level=2)  # Create an instance of the StairEnv environment for stair climbing tasks

    # wrapper_fn = wrapper.wrap_for_brax_training  # Use the standard Brax wrapper for training
    wrapper_fn = CurriculumWrapper.wrap_for_curriculum_training  # Use the custom curriculum wrapper for training


    env_cfg = env.config  # Retrieve the environment configuration
    ppo_params = {
        'action_repeat': 1,
        'batch_size': 4096,  
        'discounting': 0.995,
        'entropy_cost': 0.01,
        'episode_length': env_cfg.episode_length,
        'learning_rate': 3e-4,
        'num_envs': 4096,
        'num_evals': 5,
        'num_minibatches': 32,
        'num_updates_per_batch': 4,
        'num_timesteps': 5_000_000,
        'normalize_observations': True,
        'reward_scaling': 1.0,
        'unroll_length': 32,
        'deterministic_eval': True,
        }

    #---------- WandB logging setup ------------#
    wandb.login()
    project = "2DWalt_PPO"
    reward_config = env.reward_config
    command_config = env.command_config
    wandb_config = {
        "ppo_params": ppo_params,
        "reward_config": reward_config,
        "command_config": command_config,
        "env_config": env_cfg,
        "notes": notes,
    }
    run = wandb.init(project=project, config=wandb_config)
    
    # Visualization setup - initialize after WandB run
    render_dir = f"visualizations/{run.id}"
    html_renderer = HTML_Renderer.HTMLRenderer(env, render_dir=render_dir)


    def _to_float(value):
        """Safely converts JAX/NumPy scalars to plain Python floats for logging."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
        
    
    # Progress callback - logs metrics at each evaluation
    def progress(num_steps, metrics):
        if not hasattr(progress, "eval_counter"):
            progress.eval_counter = 0

        if progress.eval_counter == 0:
            print("Available Metric Keys:", list(metrics.keys()))

        print(f"\nEvaluation #{progress.eval_counter}:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        wandb_metrics = {k: _to_float(v) for k, v in metrics.items()}
        wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
        wandb_metrics["num_steps"] = int(num_steps)
        run.log(wandb_metrics, step=int(num_steps))
        
        progress.eval_counter += 1

    
    # Policy params callback - called periodically during training with current policy
    def policy_params_fn(num_steps, make_policy, params):
        """Callback to render visualizations during training.
        
        Args:
            num_steps: Current training step
            make_policy: Function that creates a policy from params
            params: Current network parameters
        """
        if not hasattr(policy_params_fn, "vis_counter"):
            policy_params_fn.vis_counter = 0
        
        # Render every 1,000,000 steps as requested
        if num_steps % 1_000_000 == 0 or policy_params_fn.vis_counter == 0:
            try:
                print(f"\n[Visualization] Generating policy visualization at step {num_steps}...")
                
                # Create policy from current parameters
                policy_fn = make_policy(params)
                
                # Initialize environment and get initial state
                key = jax.random.PRNGKey(42)
                state = env.reset(key)
                
                # Unroll policy trajectory
                _, trajectory = html_renderer.unroll_policy_trajectory(
                    state=state,
                    policy=policy_fn,
                    key=key,
                    num_steps=html_renderer.episode_length,
                )
                
                # Render to HTML file
                html_file = html_renderer.render_trajectory_to_html(
                    trajectory=trajectory,
                    iteration=policy_params_fn.vis_counter,
                    filename_prefix="policy_viz",
                )
                print(f"[Visualization] Saved to {html_file}")
                
                # Log HTML to WandB
                html_content = html_renderer.render_and_get_html(trajectory)
                wandb.log({
                    "policy_visualization": wandb.Html(html_content),
                    "visualization_step": num_steps,
                }, step=num_steps)
                
                policy_params_fn.vis_counter += 1
                
            except Exception as e:
                print(f"[Visualization] Warning: Failed to generate visualization at step {num_steps}: {e}")
                import traceback
                traceback.print_exc()

    
    ppo_training_params = ppo_params.copy()
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_training_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, 
            **ppo_params.network_factory
        )
    
    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
    )
    
    train_kwargs = dict(
        environment=env,
        wrap_env_fn=wrapper_fn,
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
        print("Starting PPO training from scratch...")
        make_inference_fn, params, metrics = train_fn(
            **train_kwargs
        )

    # Save the trained policy parameters and metrics
    model.save_params(save_path, params)

    run.finish()
    print(f"\nFinal metrics: {metrics}")

if __name__ == "__main__":
    main()  
