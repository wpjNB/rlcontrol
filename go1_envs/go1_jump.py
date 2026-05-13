"""Go1JumpEnv: Go1 robot obstacle jumping task."""

import numpy as np
from go1_envs.base import Go1BaseEnv

SCENE_FILE = "go1_envs/scenes/obstacle_scene.xml"
OBSTACLE_X = 2.0  # obstacle x position
OBSTACLE_TOP = 0.2  # obstacle top height (center 0.1 + half-size 0.1)


class Go1JumpEnv(Go1BaseEnv):
    """Go1 must jump over an obstacle at x=2.0m.

    Observation (39d): body_z, quat, vel, angvel, joint_pos, joint_vel, foot_contacts
    Action (12d): target joint angles
    """

    def __init__(self, render_mode=None, **kwargs):
        super().__init__(xml_file=SCENE_FILE, render_mode=render_mode, **kwargs)
        self._prev_x = 0.0
        self._crossed_obstacle = False
        self._hit_obstacle = False

    def _check_obstacle_contact(self) -> bool:
        """Check if any body part is in contact with the obstacle geom."""
        # Find the obstacle body id
        obstacle_body = None
        for b in range(self.model.nbody):
            if self.model.body(b).name == "obstacle":
                obstacle_body = b
                break
        if obstacle_body is None:
            return False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            body1 = self.model.geom_bodyid[c.geom1] if c.geom1 >= 0 else -1
            body2 = self.model.geom_bodyid[c.geom2] if c.geom2 >= 0 else -1
            if body1 == obstacle_body or body2 == obstacle_body:
                return True
        return False

    def _get_reward(self, obs, action):
        body_z = obs[0]
        body_vel_x = obs[5]
        x_pos = self.data.qpos[0]

        # Forward velocity reward
        reward = 1.0 * body_vel_x

        # Jump height reward when near obstacle
        if OBSTACLE_X - 0.5 < x_pos < OBSTACLE_X + 0.5:
            height_bonus = 10.0 * max(0, body_z - OBSTACLE_TOP)
            reward += height_bonus

        # Obstacle collision penalty
        if self._check_obstacle_contact():
            reward -= 100.0
            self._hit_obstacle = True

        # Successfully crossed obstacle
        if x_pos > OBSTACLE_X + 0.3 and not self._crossed_obstacle and not self._hit_obstacle:
            reward += 50.0
            self._crossed_obstacle = True

        return float(reward)

    def _is_terminated(self):
        body_z = self._get_body_z()

        # Fell down
        if body_z < 0.15:
            return True

        # Hit obstacle
        if self._hit_obstacle:
            return True

        # Reached goal
        x_pos = self.data.qpos[0]
        if x_pos > 5.0:
            return True

        return False

    def _get_info(self):
        return {
            "body_z": self._get_body_z(),
            "x_pos": float(self.data.qpos[0]),
            "crossed_obstacle": self._crossed_obstacle,
            "hit_obstacle": self._hit_obstacle,
        }

    def reset(self, seed=None, options=None):
        self._prev_x = 0.0
        self._crossed_obstacle = False
        self._hit_obstacle = False
        return super().reset(seed=seed, options=options)
