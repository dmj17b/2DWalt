# Policy Visualization System

HTML-based interactive visualization of policy behavior during Brax PPO training, with automatic logging to WandB for remote monitoring.

## Quick Start

```python
from visualization.HTML_Renderer import HTMLRenderer
import jax

# Initialize renderer
renderer = HTMLRenderer(env, render_dir="visualizations")

# Unroll policy trajectory
key = jax.random.PRNGKey(0)
state = env.reset(key)
_, trajectory = renderer.unroll_policy_trajectory(
    state=state,
    policy=policy_fn,
    key=key,
    num_steps=1000
)

# Render to HTML file
html_file = renderer.render_trajectory_to_html(
    trajectory=trajectory,
    iteration=0,
    filename_prefix="policy_viz"
)

# Get HTML string for WandB logging
html_str = renderer.render_and_get_html(trajectory)
wandb.log({"visualization": wandb.Html(html_str)})
```

## Integration with Training

The system is fully integrated into `train/trainVis.py`:

```python
python train/trainVis.py
```

This automatically:
- Renders visualizations every 1,000,000 training steps
- Saves HTML files to `visualizations/{run_id}/`
- Logs visualizations to WandB
- Continues training even if visualization fails

## API Reference

### HTMLRenderer Class

#### `__init__(env, render_dir=None, episode_length=None)`
Initialize the HTML renderer.

**Args:**
- `env` (Env): Brax environment with MuJoCo backend
- `render_dir` (str, optional): Directory to save HTML files. Defaults to `./visualizations`
- `episode_length` (int, optional): Episode length for rendering. Auto-detected if not provided

#### `unroll_policy_trajectory(state, policy, key, num_steps=None)`
Unroll a policy trajectory.

**Args:**
- `state` (State): Initial environment state
- `policy` (PolicyFn): Policy function that takes `(obs, key)` and returns `(actions, data)`
- `key` (jax.Array): JAX random key
- `num_steps` (int, optional): Number of steps. Defaults to episode_length

**Returns:**
- Tuple of (final_state, trajectory_data)
- trajectory_data is a tuple of (qpos, xpos, xquat) JAX arrays

#### `render_trajectory_to_html(trajectory, iteration, filename_prefix)`
Render a trajectory to an HTML file.

**Args:**
- `trajectory`: Tuple of (qpos, xpos, xquat) from `unroll_policy_trajectory()`
- `iteration` (int): Iteration number for naming
- `filename_prefix` (str): Prefix for filename (default: "trajectory")

**Returns:**
- str: Path to the generated HTML file

#### `render_and_get_html(trajectory)`
Render a trajectory and return HTML string.

**Args:**
- `trajectory`: Tuple of (qpos, xpos, xquat) from `unroll_policy_trajectory()`

**Returns:**
- str: HTML string ready for WandB logging

## Configuration

### Render Frequency
Edit `train/trainVis.py`, around line 126:

```python
# Change from:
if num_steps % 1_000_000 == 0:
# To:
if num_steps % 500_000 == 0:  # Render every 500k steps
```

### Trajectory Length
Edit `train/trainVis.py`, around line 129:

```python
# Default: use full episode length
num_steps=html_renderer.episode_length

# Or use custom length:
num_steps=100  # Faster rendering
```

## Features

✓ **Interactive Rendering**: Drag, pan, zoom, rotate 3D visualization
✓ **JAX-Accelerated**: Uses JAX scan for efficient trajectory collection
✓ **WandB Integration**: Automatic logging of visualizations
✓ **Local Storage**: Saves HTML files for offline review
✓ **Error Resilient**: Visualization failures don't stop training
✓ **Configurable**: Rendering frequency and trajectory length customizable
✓ **Production Ready**: Type hints, documentation, error handling

## Requirements

- `brax >= 0.10.0`
- `mujoco >= 3.0.0`
- `jax >= 0.4.0`
- `wandb >= 0.13.0`
- `numpy`

All included in `requirements.txt`

## WandB Integration

Visualizations are automatically logged to WandB. View in WandB:
1. Open your project page
2. Click on the run
3. Scroll to "policy_visualization" chart
4. Click to view interactive HTML

## Troubleshooting

### Visualizations not appearing in WandB
- Verify WandB login: `wandb login`
- Check that `policy_params_fn` is passed to `ppo.train()`
- Look for `[Visualization]` messages in console logs

### HTML files not saving
- Check write permissions in render directory
- Verify disk space available
- Check console for error messages
