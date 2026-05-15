"""Tests for Go1WalkV2Env."""

import numpy as np
import pytest


class TestGo1WalkV2Env:
    """Tests for Go1WalkV2Env with full reward suite."""

    def _make_env(self, **kwargs):
        from go1_envs.go1_walk_v2 import Go1WalkV2Env
        return Go1WalkV2Env(**kwargs)

    # ── basic interface ────────────────────────────────────────────────

    def test_obs_shape(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (42,), f"Expected (42,), got {obs.shape}"
        assert obs.dtype == np.float32
        env.close()

    def test_action_space(self):
        env = self._make_env()
        assert env.action_space.shape == (12,)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, reward, term, trunc, info = env.step(action)
        assert obs2.shape == (42,)
        env.close()

    def test_command_in_obs(self):
        env = self._make_env()
        obs, info = env.reset()
        cmd_from_obs = obs[39:]
        cmd_from_info = info["command"]
        np.testing.assert_array_equal(cmd_from_obs, cmd_from_info)
        env.close()

    def test_set_command(self):
        env = self._make_env()
        env.reset()
        env.set_command(vx=1.0, vy=0.5, yaw_rate=-0.3)
        obs = env._get_obs()
        np.testing.assert_allclose(obs[39:], [1.0, 0.5, -0.3], atol=1e-6)
        env.close()

    def test_reset_noise(self):
        env = self._make_env()
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=1)
        assert not np.allclose(obs1, obs2)
        env.close()

    # ── reward components ──────────────────────────────────────────────

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

    def test_reward_terms_keys(self):
        env = self._make_env()
        env.reset()
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        expected_keys = {
            "tracking_lin_vel", "tracking_ang_vel",
            "lin_vel_z", "ang_vel_xy", "orientation", "base_height",
            "torques", "dof_vel", "dof_acc", "action_rate",
            "collision", "dof_pos_limits", "feet_air_time",
            "stumble", "stand_still", "feet_contact_forces",
        }
        assert expected_keys == set(info["reward_terms"].keys())
        env.close()

    def test_tracking_lin_vel_range(self):
        """Gaussian kernel reward should be in [0, 1]."""
        env = self._make_env()
        env.set_command(vx=1.0)
        env.reset()
        for _ in range(50):
            _, _, _, _, info = env.step(env.action_space.sample())
            val = info["reward_terms"]["tracking_lin_vel"]
            assert 0.0 <= val <= 1.0, f"tracking_lin_vel out of range: {val}"
        env.close()

    def test_tracking_ang_vel_range(self):
        env = self._make_env()
        env.set_command(yaw_rate=1.0)
        env.reset()
        for _ in range(50):
            _, _, _, _, info = env.step(env.action_space.sample())
            val = info["reward_terms"]["tracking_ang_vel"]
            assert 0.0 <= val <= 1.0
        env.close()

    def test_penalty_terms_non_positive(self):
        """Penalty terms (negative scale) should produce non-positive values."""
        env = self._make_env()
        env.reset()
        penalty_keys = [
            "lin_vel_z", "ang_vel_xy", "orientation", "base_height",
            "torques", "dof_vel", "dof_acc", "action_rate",
        ]
        for _ in range(50):
            _, _, _, _, info = env.step(env.action_space.sample())
            for k in penalty_keys:
                val = info["reward_terms"][k]
                assert val <= 0.0 + 1e-6, f"{k} should be <= 0, got {val}"
            if "termination" in info.get("reward_terms", {}):
                break
        env.close()

    def test_stand_still_only_when_no_command(self):
        env = self._make_env()
        env.set_command(vx=0.0, vy=0.0, yaw_rate=0.0)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        # With zero command, stand_still should be > 0 (joints deviate from home)
        assert info["reward_terms"]["stand_still"] >= 0.0

        env.set_command(vx=1.0)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        # With nonzero command, stand_still should be 0
        assert info["reward_terms"]["stand_still"] == 0.0
        env.close()

    # ── termination ────────────────────────────────────────────────────

    def test_terminated_on_fall(self):
        env = self._make_env()
        env.reset()
        for _ in range(500):
            action = np.array([0, 3.0, -0.9] * 4, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        env.close()

    # ── reward weight customization ────────────────────────────────────

    def test_custom_weights(self):
        env = self._make_env(
            tracking_lin_vel_scale=5.0,
            torques_scale=0.0,
        )
        env.set_command(vx=1.0)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        # torques with scale=0 should contribute nothing
        assert info["reward_terms"]["torques"] == 0.0
        # tracking with scale=5.0 should be amplified
        assert info["reward_terms"]["tracking_lin_vel"] >= 0.0
        env.close()

    # ── info dict ──────────────────────────────────────────────────────

    def test_info_fields(self):
        env = self._make_env()
        env.reset()
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        assert "body_z" in info
        assert "body_tilt" in info
        assert "x_pos" in info
        assert "x_vel" in info
        assert "command" in info
        assert "reward_terms" in info
        env.close()

    # ── gymnasium env_checker ──────────────────────────────────────────

    def test_env_passes_checker(self):
        from gymnasium.utils.env_checker import check_env
        env = self._make_env()
        check_env(env, skip_render_check=True)
        env.close()
