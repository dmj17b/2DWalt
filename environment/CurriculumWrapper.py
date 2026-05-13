from brax.envs.base import Env, Wrapper
import jax
import jax.numpy as jp

class CurriculumWrapper(Wrapper):
    """Automatically resets the environment while ferrying the curriculum state."""
    
    def step(self, state, action):
        # 1. Calculate the standard forward transition
        next_state = self.env.step(state, action)
        
        # 2. Generate a potential reset state USING the newly achieved challenge level
        rng, rng_reset = jax.random.split(next_state.info["rng"])
        reset_state = self.env.level_reset(rng_reset, next_state.info["challenge_level"])
        
        # 3. Mathematically route the state PyTree based on the done flag
        def where_done(reset_val, next_val):
            done = next_state.done
            if done.shape:
                done = jp.reshape(done, [next_val.shape[0]] + [1] * (len(next_val.shape) - 1))
            return jp.where(done, reset_val, next_val)
            
        # Selects reset_state if done==1.0, otherwise keeps next_state
        state = jax.tree_util.tree_map(where_done, reset_state, next_state)
        return state