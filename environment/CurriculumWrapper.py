from brax.envs.base import Env, Wrapper
from mujoco_playground._src import wrapper
import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env
from typing import Any, Callable, List, Optional, Sequence, Tuple
from mujoco import mjx
from brax.envs.wrappers import training as brax_training

def wrap_for_curriculum_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 10000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    full_reset: bool = False,
) -> Wrapper:
    """Wrapper for curriculum training that automatically resets the environment while ferrying the curriculum state."""
    # env = CurriculumWrapper(env)  # Wrap the environment with the curriculum wrapper to enable automatic resets and curriculum state management

    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)

    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = CurriculumResetWrapper(env)
    return env




class CurriculumResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done.

  Attributes:
    env: The wrapped environment.
  """

  def __init__(self, env: Any):
    super().__init__(env)
    self._info_key = 'CurriculumResetWrapper'

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng_key = jax.vmap(jax.random.split)(rng)
    rng, key = rng_key[..., 0], rng_key[..., 1]
    state = self.env.reset(key)
    state.info[f'{self._info_key}_first_data'] = state.data
    state.info[f'{self._info_key}_first_obs'] = state.obs
    state.info[f'{self._info_key}_rng'] = rng
    state.info[f'{self._info_key}_done_count'] = jp.zeros(
        key.shape[:-1], dtype=int
    )
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # grab the reset state.
    reset_state = None
    rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
    reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]

    reset_data = jax.vmap(self.env.level_reset)(reset_key, state.info["challenge_level"]).data
    reset_obs = jax.vmap(self.env.level_reset)(reset_key, state.info["challenge_level"]).obs

    if 'steps' in state.info:
      # reset steps to 0 if done.
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)

    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape and done.shape[0] != x.shape[0]:
        return y
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)

    data = jax.tree.map(where_done, reset_data, state.data)
    obs = jax.tree.map(where_done, reset_obs, state.obs)

    next_info = state.info
    done_count_key = f'{self._info_key}_done_count'
    

    next_info[done_count_key] += state.done.astype(int)
    next_info[f'{self._info_key}_rng'] = reset_rng

    return state.replace(data=data, obs=obs, info=next_info)
  

class CurriculumWrapper(Wrapper):
    """Automatically resets the environment while ferrying the curriculum state."""

    def step(self, state, action):
        # 1. Calculate the standard forward transition from the base environment
        next_state = self.env.step(state, action)
        
        rng, rng_reset = jax.random.split(next_state.info["rng"])
        reset_state = self.env.level_reset(rng_reset, next_state.info["challenge_level"])


        def where_done(reset_val, next_val):
            done = next_state.done
            if done.shape:
                done = jp.reshape(done, [next_val.shape[0]] + [1] * (len(next_val.shape) - 1))
            return jp.where(done, reset_val, next_val)
            
        new_data = jax.tree_util.tree_map(where_done, reset_state.data, next_state.data)
        new_obs = jax.tree_util.tree_map(where_done, reset_state.obs, next_state.obs)
        new_info = jax.tree_util.tree_map(where_done, reset_state.info, next_state.info)

        # Reconstruct the state PyTree
        state = next_state.replace(
            data=new_data,
            obs=new_obs,
            info=new_info
        )
        return state
