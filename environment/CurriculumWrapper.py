from brax.envs.base import Wrapper, State, Env
from mujoco_playground._src import wrapper
import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env
from typing import Callable, Optional, Tuple
from mujoco import mjx
from brax.envs.wrappers import training as brax_training

# class CurriculumWrapper(Wrapper):
#     """Automatically resets the environment while ferrying the curriculum state."""
    
#     def step(self, state, action):
#         # 1. Calculate the standard forward transition
#         next_state = self.env.step(state, action)
        
#         # 2. Generate a potential reset state USING the newly achieved challenge level
#         rng, rng_reset = jax.random.split(next_state.info["rng"])
#         reset_state = self.env.level_reset(rng_reset, next_state.info["challenge_level"])
        
#         # 3. Mathematically route the state PyTree based on the done flag
#         def where_done(reset_val, next_val):
#             done = next_state.done
#             if done.shape:
#                 done = jp.reshape(done, [next_val.shape[0]] + [1] * (len(next_val.shape) - 1))
#             return jp.where(done, reset_val, next_val)
            
#         # Selects reset_state if done==1.0, otherwise keeps next_state
#         state = jax.tree_util.tree_map(where_done, reset_state, next_state)
#         return state
class CurriculumWrapper(Wrapper):
    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["steps"] = jp.zeros_like(state.done, dtype=jp.int32)
        return state

    def step(self, state, action):
        next_state = self.env.step(state, action)

        rng, rng_reset = jax.random.split(next_state.info["rng"])
        reset_state = self.env.level_reset(rng_reset, next_state.info["challenge_level"])

        done = next_state.done

        def blend(reset_val, next_val):
            return jp.where(done, reset_val, next_val)

        data = jax.tree_util.tree_map(blend, reset_state.data, next_state.data)
        obs = jax.tree_util.tree_map(blend, reset_state.obs, next_state.obs)
        info = jax.tree_util.tree_map(blend, reset_state.info, next_state.info)
        info["steps"] = jp.where(done, jp.zeros_like(info["steps"]), info["steps"])

        return next_state.replace(
            data=data,
            obs=obs,
            reward=blend(reset_state.reward, next_state.reward),
            done=blend(reset_state.done, next_state.done),
            info=info,
            metrics=next_state.metrics,
        )

def wrap_for_curriculum_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    full_reset: bool = False,
) -> Wrapper:
    """Wrap an env for curriculum training without episode truncation."""
    env = CurriculumWrapper(env)  # Wrap the environment with the curriculum wrapper to enable automatic resets and curriculum state management
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = wrapper.BraxAutoResetWrapper(env, full_reset=full_reset)

    return env

class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    # Keep separate record of episode done as state.info['done'] can be erased
    # by AutoResetWrapper
    state.info['episode_done'] = jp.zeros(rng.shape[:-1])
    episode_metrics = dict()
    episode_metrics['sum_reward'] = jp.zeros(rng.shape[:-1])
    episode_metrics['length'] = jp.zeros(rng.shape[:-1])
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
    state.info['episode_metrics'] = episode_metrics
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps

    # Aggregate state metrics into episode metrics
    prev_done = state.info['episode_done']
    state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
    state.info['episode_metrics']['sum_reward'] += jp.sum(rewards, axis=0)
    state.info['episode_metrics']['length'] *= (1 - prev_done)
    state.info['episode_metrics']['length'] += self.action_repeat
    for metric_name in state.metrics.keys():
      if metric_name != 'reward':
        state.info['episode_metrics'][metric_name] *= (1 - prev_done)
        state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
    state.info['episode_done'] = done
    return state.replace(done=done)
