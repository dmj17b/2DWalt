import EnvWalt2D
from datetime import datetime
from IPython.display import HTML, clear_output, display
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from mujoco_playground import wrapper
import functools
import mediapy as media

import matplotlib.pyplot as plt
import jax

from mujoco_playground.config import dm_control_suite_params

env = EnvWalt2D.EnvWalt2D()  # Create an instance of the EnvWalt2D environment
env_cfg = env.config  # Retrieve the environment configuration
ppo_params = {
    'action_repeat': 1,
    'batch_size': 64,
    'discounting': 0.995,
    'entropy_cost': 0.01,
    'episode_length': 1000,
    'learning_rate': 3e-4,
    'num_envs': 1024,
    'num_evals': 100,  # More frequent progress updates (every ~100k steps)
    'num_minibatches': 32,
    'num_updates_per_batch': 16,
    'num_timesteps': 100000000,  
    'normalize_observations': True,
    'reward_scaling': 1.0,
    'unroll_length': 30,
    }


x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]
training_start_time = None
fig, ax = None, None  # Global figure and axis for updating

def progress(num_steps, metrics):
    global training_start_time, fig, ax
    
    # Record training start time on first progress call
    if training_start_time is None:
        training_start_time = datetime.now()
        times.append(training_start_time)
        # Create figure once on first call
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))
    
    times.append(datetime.now())
    x_data.append(num_steps)
    
    # Use the correct episode reward metrics from eval
    y_data.append(metrics['eval/episode_reward'])
    y_dataerr.append(metrics.get('eval/episode_reward_std', 0.0))

    # Update the existing plot
    ax.clear()
    ax.set_xlim([0, ppo_params["num_timesteps"] * 1.25])
    ax.set_ylim([-200, 1100])
    ax.set_xlabel("# environment steps")
    ax.set_ylabel("reward per episode")
    ax.set_title(f"Steps: {num_steps}, Reward: {y_data[-1]:.3f}")
    ax.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    ax.grid(True, alpha=0.3)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)  # Allow plot to update
    
    # Print progress to console
    elapsed = (datetime.now() - training_start_time).total_seconds()
    print(f"Steps: {num_steps}/{ppo_params['num_timesteps']}, "
          f"Reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}, "
          f"Time elapsed: {elapsed:.1f}s")
    
    # Print all metrics for debugging
    print(f"  Available metrics: {list(metrics.keys())}")
    print(f"  Progress callback #{len(x_data)}")
    


ppo_training_params = dict(ppo_params)

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

print("\nTraining completed!")
print(f"Time to JIT compile: {times[0] if len(times) > 0 else 'N/A'}")
if training_start_time:
    total_training_time = (datetime.now() - training_start_time).total_seconds()
    print(f"Time to train: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Progress callback was called {len(x_data)} times")
else:
    print("WARNING: Progress callback was never called!")
    
print(f"\nFinal metrics: {metrics}")



jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1

for _ in range(n_episodes):
  state = jit_reset(rng)
  rollout.append(state)
  for i in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)

render_every = 1
frames = env.render(rollout[::render_every])
rewards = [s.reward for s in rollout]
media.show_video(frames, fps=1.0 / env.dt / render_every)
media.write_video("walt2d_ppo.mp4", frames, fps=1.0 / env.dt / render_every)