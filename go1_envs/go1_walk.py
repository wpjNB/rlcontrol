"""Go1WalkEnv: Go1 robot forward walking task."""

import numpy as np
from go1_envs.base import Go1BaseEnv

SCENE_FILE = "go1_envs/scenes/flat_scene.xml"


class Go1WalkEnv(Go1BaseEnv):
    """Go1 must learn to walk forward along the x-axis.

    Observation (39d): body_z, quat, vel, angvel, joint_pos, joint_vel, foot_contacts
    Action (12d): target joint angles
    """

    def __init__(self, render_mode=None, **kwargs):
        super().__init__(xml_file=SCENE_FILE, render_mode=render_mode, **kwargs)

    def _get_reward(self, obs, action):
        body_vel_x = obs[5]  # x-component of body linear velocity
        body_vel_y = obs[6]  # y-component (lateral)

        # Forward velocity reward
        reward = 1.0 * body_vel_x

        # Alive reward
        reward += 1.0

        # Control cost
        reward -= 0.05 * np.sum(action ** 2)

        # Lateral velocity penalty
        reward -= 0.1 * body_vel_y ** 2

        return float(reward)

    def _is_terminated(self):
        body_z = self._get_body_z()

        # Fell down
        if body_z < 0.15:
            return True

        # Drifted too far laterally
        y_pos = self.data.qpos[1]
        if abs(y_pos) > 2.0:
            return True

        # Reached goal
        x_pos = self.data.qpos[0]
        if x_pos > 20.0:
            return True

        return False

    def _get_info(self):
        return {
            "body_z": self._get_body_z(),
            "x_pos": float(self.data.qpos[0]),
            "y_pos": float(self.data.qpos[1]),
            "x_vel": float(self.data.qvel[0]),
        }
