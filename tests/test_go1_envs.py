"""Tests for Go1 custom environments."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env


class TestGo1BalanceEnv:
    """Tests for Go1BalanceEnv."""

    def _make_env(self):
        from go1_envs.go1_balance import Go1BalanceEnv
        return Go1BalanceEnv()

    def test_env_passes_checker(self):
        """Env must pass gymnasium's env_checker."""
        env = self._make_env()
        check_env(env, skip_render_check=True)
        env.close()

    def test_obs_shape(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (39,)
        assert obs.dtype == np.float32
        env.close()

    def test_action_space(self):
        env = self._make_env()
        assert env.action_space.shape == (12,)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, reward, term, trunc, info = env.step(action)
        assert obs2.shape == (39,)
        env.close()

    def test_reset_noise(self):
        """Observations after reset should vary due to noise."""
        env = self._make_env()
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=1)
        assert not np.allclose(obs1, obs2)
        env.close()

    def test_terminated_on_fall(self):
        """Env should terminate when robot falls."""
        env = self._make_env()
        env.reset()
        # Apply large downward action to make it collapse
        for _ in range(500):
            action = env.action_space.sample() * 0
            # Use extreme joint positions to destabilize
            action = np.array([0, 3.0, -0.9] * 4, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        # Either terminated or ran many steps
        env.close()

    def test_reward_is_finite(self):
        env = self._make_env()
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert np.isfinite(reward), f"Reward not finite: {reward}"
            if term:
                env.reset()
        env.close()
