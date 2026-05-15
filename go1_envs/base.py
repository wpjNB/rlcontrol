"""Go1BaseEnv: Base class for Go1 quadruped robot environments."""

import os
from contextlib import contextmanager
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

@contextmanager
def _noop_ctx():
    yield

# Foot geom IDs in the Go1 model (verified)
FOOT_GEOM_NAMES = ["FR", "FL", "RR", "RL"]

# Home keyframe joint angles (hip, thigh, knee) per leg
HOME_QPOS = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
HOME_Z = 0.27

# action observe space 几乎不变，reward和termination根据不同环境（你想要功能）定义
class Go1BaseEnv(gym.Env):
    """Base environment for Go1 quadruped robot tasks.

    Subclasses must implement:
        - _get_reward(obs, action) -> float
        - _is_terminated() -> bool
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        xml_file: str,
        frame_skip: int = 25,
        reset_noise_scale: float = 0.1,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.reset_noise_scale = reset_noise_scale

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # Renderer for human/rgb_array modes
        self._renderer = None
        self._viewer = None

        # Cache foot geom IDs
        self._foot_geom_ids = []
        for name in FOOT_GEOM_NAMES:
            for i in range(self.model.ngeom):
                if self.model.geom(i).name == name:
                    self._foot_geom_ids.append(i)
                    break

        # Action space: 12 joint target angles
        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0].astype(np.float32),
            high=self.model.actuator_ctrlrange[:, 1].astype(np.float32),
            dtype=np.float32,
        )

        # Observation space (39 dims):
        # [body_z(1), body_quat(4), body_vel(3), body_angvel(3),
        #  joint_pos(12), joint_vel(12), foot_contacts(4)]
        obs_dim = 1 + 4 + 3 + 3 + 12 + 12 + 4
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current simulation state."""
        data = self.data
        body_z = np.array([data.qpos[2]], dtype=np.float32)
        body_quat = data.qpos[3:7].astype(np.float32)  # wxyz
        body_vel = data.qvel[0:3].astype(np.float32)  # linear velocity
        body_angvel = data.qvel[3:6].astype(np.float32)  # angular velocity
        joint_pos = data.qpos[7:19].astype(np.float32)  # 12 joint angles
        joint_vel = data.qvel[6:18].astype(np.float32)  # 12 joint velocities
        foot_contacts = self._get_foot_contacts().astype(np.float32)  # 4 binary

        return np.concatenate([
            body_z, body_quat, body_vel, body_angvel,
            joint_pos, joint_vel, foot_contacts,
        ])

    def _get_foot_contacts(self) -> np.ndarray:
        """Check which feet are in contact with the ground."""
        contacts = np.zeros(4, dtype=np.float32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for j, foot_id in enumerate(self._foot_geom_ids):
                if c.geom1 == foot_id or c.geom2 == foot_id:
                    contacts[j] = 1.0
                    break
        return contacts

    def _get_body_z(self) -> float:
        """Return trunk height."""
        return float(self.data.qpos[2])

    def _get_body_tilt(self) -> float:
        """Return tilt angle (radians) from upright. 0 = perfectly upright."""
        quat = self.data.qpos[3:7]
        # Convert quaternion to rotation matrix, extract z-axis component
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        # z-axis of body frame in world frame
        z_axis = rot[:, 2]
        # Angle from vertical: arccos(z_axis . [0,0,1])
        cos_angle = np.clip(z_axis[2], -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def _get_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Subclasses must implement this."""
        raise NotImplementedError

    def _is_terminated(self) -> bool:
        """Subclasses must implement this."""
        raise NotImplementedError

    def _get_info(self) -> dict:
        """Optional override for extra logging info."""
        return {}

    def step(self, action: np.ndarray):
        # Apply action
        self.data.ctrl[:] = action

        # Step simulation frame_skip times (lock viewer to avoid data race)
        ctx = self._viewer.lock() if self._viewer is not None else _noop_ctx()
        with ctx:
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._get_reward(obs, action)
        terminated = self._is_terminated()
        truncated = False  # handled by TimeLimit wrapper
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Lock viewer to avoid data race during reset
        ctx = self._viewer.lock() if self._viewer is not None else _noop_ctx()
        with ctx:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

            # Add noise to joint positions (not to body pose)
            noise = self.np_random.uniform(
                -self.reset_noise_scale, self.reset_noise_scale, size=12
            ).astype(np.float64)
            self.data.qpos[7:19] += noise

            # Forward dynamics to compute contacts etc.
            mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            if self._viewer.is_running():
                self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            try:
                if self._viewer.is_running():
                    self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
