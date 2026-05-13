"""Go1BalanceEnv: Go1 robot balance standing task."""

import numpy as np
from go1_envs.base import Go1BaseEnv, HOME_Z

SCENE_FILE = "go1_envs/scenes/flat_scene.xml"


class Go1BalanceEnv(Go1BaseEnv):
    """Go1 must maintain standing posture against random perturbations.

    Observation (39d): body_z, quat, vel, angvel, joint_pos, joint_vel, foot_contacts
    Action (12d): target joint angles
    """

    def __init__(self, render_mode=None, **kwargs):
        super().__init__(xml_file=SCENE_FILE, render_mode=render_mode, **kwargs)
        self._step_count = 0
        self._perturb_remaining = 0
        self._perturb_force = np.zeros(6)

    def _get_reward(self, obs, action):
        body_z = obs[0]
        body_quat = obs[1:5]
        joint_pos = obs[17:29]

        # Alive reward
        reward = 1.0

        # Tilt penalty: penalize deviation from upright
        # Simple proxy: quaternion w component should be close to 1 for upright
        tilt_penalty = 0.5 * (1.0 - body_quat[0]) ** 2
        reward -= tilt_penalty

        # Action cost: penalize large actions
        action_cost = 0.1 * np.sum(action ** 2)
        reward -= action_cost

        return float(reward)

    def _is_terminated(self):
        body_z = self._get_body_z()
        tilt = self._get_body_tilt()

        # Fell down
        if body_z < 0.15:
            return True
        # Tilted too much (> 45 degrees)
        if tilt > np.pi / 4:
            return True
        return False

    def _get_info(self):
        return {
            "body_z": self._get_body_z(),
            "body_tilt": self._get_body_tilt(),
            "foot_contacts": self._get_foot_contacts().tolist(),
        }

    def _apply_perturbation(self):
        """Apply random horizontal force perturbation."""
        self._step_count += 1

        # If currently applying perturbation
        if self._perturb_remaining > 0:
            self.data.xfrc_applied[1, :3] = self._perturb_force[:3]  # trunk body
            self._perturb_remaining -= 1
            return

        # Clear any previous perturbation
        self.data.xfrc_applied[1, :3] = np.zeros(3)

        # Random chance to start a new perturbation
        if self._step_count > 100 and self.np_random.random() < 0.01:
            force_mag = self.np_random.uniform(10, 30)
            direction = self.np_random.uniform(-np.pi, np.pi)
            self._perturb_force = np.array([
                force_mag * np.cos(direction),
                force_mag * np.sin(direction),
                0, 0, 0, 0,
            ])
            self._perturb_remaining = 10

    def step(self, action):
        self._apply_perturbation()
        return super().step(action)

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._perturb_remaining = 0
        self._perturb_force = np.zeros(6)
        self.data.xfrc_applied[:] = 0
        return super().reset(seed=seed, options=options)
