"""Go1WalkV2Env: Go1 walking with full reward suite from IsaacGym/Legged Gym."""

import numpy as np
import mujoco
from go1_envs.base import Go1BaseEnv, HOME_QPOS

SCENE_FILE = "go1_envs/scenes/flat_scene.xml"

# Model structure
FOOT_BODY_NAMES = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
NON_FOOT_BODY_NAMES = [
    "trunk", "FR_hip", "FR_thigh", "FL_hip", "FL_thigh",
    "RR_hip", "RR_thigh", "RL_hip", "RL_thigh",
]
TARGET_BASE_HEIGHT = 0.30


class Go1WalkV2Env(Go1BaseEnv):
    """Go1 walking environment with comprehensive reward system.

    Observation (42d):
        body_z(1), quat(4), body_vel(3), body_angvel(3),
        joint_pos(12), joint_vel(12), foot_contacts(4),
        commanded_vel(3)

    Action (12d): target joint angles
    """

    def __init__(
        self,
        render_mode=None,
        # reward scales
        tracking_lin_vel_scale=1.0,
        tracking_ang_vel_scale=0.5,
        lin_vel_z_scale=-2.0,
        ang_vel_xy_scale=-0.05,
        orientation_scale=-1.0,
        base_height_scale=-30.0,
        torques_scale=-0.0002,
        dof_vel_scale=-0.001,
        dof_acc_scale=-2.5e-7,
        action_rate_scale=-0.01,
        collision_scale=-1.0,
        termination_scale=-2.0,
        dof_pos_limits_scale=-10.0,
        feet_air_time_scale=1.0,
        stumble_scale=-2.0,
        stand_still_scale=-1.0,
        feet_contact_forces_scale=-0.01,
        # tracking sigma
        tracking_sigma=0.25,
        **kwargs,
    ):
        super().__init__(xml_file=SCENE_FILE, render_mode=render_mode, **kwargs)

        # Commanded velocity (vx, vy, yaw_rate)
        self._command = np.zeros(3, dtype=np.float32)

        # Cache body IDs
        self._foot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in FOOT_BODY_NAMES
        ]
        self._non_foot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in NON_FOOT_BODY_NAMES
        ]

        # Joint limits from model
        self._joint_range = self.model.jnt_range[1:13].copy()  # skip free joint

        # State tracking
        self._prev_action = np.zeros(12, dtype=np.float32)
        self._prev_joint_vel = np.zeros(12, dtype=np.float32)
        self._feet_air_time = np.zeros(4, dtype=np.float32)
        self._last_contact = np.zeros(4, dtype=bool)

        # Reward scales
        self._scales = dict(
            tracking_lin_vel=tracking_lin_vel_scale,
            tracking_ang_vel=tracking_ang_vel_scale,
            lin_vel_z=lin_vel_z_scale,
            ang_vel_xy=ang_vel_xy_scale,
            orientation=orientation_scale,
            base_height=base_height_scale,
            torques=torques_scale,
            dof_vel=dof_vel_scale,
            dof_acc=dof_acc_scale,
            action_rate=action_rate_scale,
            collision=collision_scale,
            termination=termination_scale,
            dof_pos_limits=dof_pos_limits_scale,
            feet_air_time=feet_air_time_scale,
            stumble=stumble_scale,
            stand_still=stand_still_scale,
            feet_contact_forces=feet_contact_forces_scale,
        )
        self._tracking_sigma = tracking_sigma

        # Override obs space: 39 + 3 (commanded_vel)
        from gymnasium import spaces
        obs_dim = 42
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._step_count = 0

    def _get_obs(self):
        base_obs = super()._get_obs()  # 39d
        cmd = self._command.astype(np.float32)
        return np.concatenate([base_obs, cmd])

    def set_command(self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0):
        self._command[:] = [vx, vy, yaw_rate]

    # ── reward components ──────────────────────────────────────────────

    def _reward_tracking_lin_vel(self):
        vel_x, vel_y = self.data.qvel[0], self.data.qvel[1]
        cmd_vx, cmd_vy = self._command[0], self._command[1]
        err = (vel_x - cmd_vx) ** 2 + (vel_y - cmd_vy) ** 2
        return float(np.exp(-err / self._tracking_sigma))

    def _reward_tracking_ang_vel(self):
        yaw_rate = self.data.qvel[5]
        cmd_yaw = self._command[2]
        err = (yaw_rate - cmd_yaw) ** 2
        return float(np.exp(-err / self._tracking_sigma))

    def _reward_lin_vel_z(self):
        return float(self.data.qvel[2] ** 2)

    def _reward_ang_vel_xy(self):
        return float(np.sum(self.data.qvel[3:5] ** 2))

    def _reward_orientation(self):
        quat = self.data.qpos[3:7]
        gravity = np.array([0.0, 0.0, -1.0])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        projected = rot.T @ gravity
        return float(np.sum(projected[:2] ** 2))

    def _reward_base_height(self):
        return float((self.data.qpos[2] - TARGET_BASE_HEIGHT) ** 2)

    def _reward_torques(self):
        return float(np.sum(self.data.ctrl ** 2))

    def _reward_dof_vel(self):
        joint_vel = self.data.qvel[6:18]
        return float(np.sum(joint_vel ** 2))

    def _reward_dof_acc(self):
        joint_vel = self.data.qvel[6:18]
        acc = (joint_vel - self._prev_joint_vel) / (self.frame_skip * self.model.opt.timestep)
        return float(np.sum(acc ** 2))

    def _reward_action_rate(self):
        return float(np.sum((self._prev_action - self._last_action) ** 2))

    def _reward_collision(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for bid in self._non_foot_body_ids:
                geom_start = self.model.body_geomadr[bid]
                geom_count = self.model.body_geomnum[bid]
                for g in range(geom_start, geom_start + geom_count):
                    if c.geom1 == g or c.geom2 == g:
                        return 1.0
        return 0.0

    def _reward_termination(self):
        return 0.0  # placeholder, set externally

    def _reward_dof_pos_limits(self):
        joint_pos = self.data.qpos[7:19]
        low = self._joint_range[:, 0]
        high = self._joint_range[:, 1]
        violation = np.maximum(low - joint_pos, 0) + np.maximum(joint_pos - high, 0)
        return float(np.sum(violation))

    def _reward_feet_air_time(self):
        contact = self._get_foot_contacts()
        reward = 0.0
        for i in range(4):
            if contact[i] > 0.5 and not self._last_contact[i]:
                # just landed: reward air time, capped at 0.5s
                reward += min(self._feet_air_time[i], 0.5)
                self._feet_air_time[i] = 0.0
            elif contact[i] < 0.5:
                dt = self.frame_skip * self.model.opt.timestep
                self._feet_air_time[i] += dt
        self._last_contact = contact > 0.5
        return reward

    def _reward_stumble(self):
        contact = self._get_foot_contacts()
        for i in range(4):
            if contact[i] > 0.5:
                force = np.zeros(6)
                for j in range(self.data.ncon):
                    c = self.data.contact[j]
                    foot_id = self._foot_geom_ids[i]
                    if c.geom1 == foot_id or c.geom2 == foot_id:
                        mujoco.mj_contactForce(self.model, self.data, j, force)
                        if np.linalg.norm(force[:3]) > 500.0:
                            return 1.0
        return 0.0

    def _reward_stand_still(self):
        if np.linalg.norm(self._command) < 0.01:
            return float(np.sum((self.data.qpos[7:19] - HOME_QPOS) ** 2))
        return 0.0

    def _reward_feet_contact_forces(self):
        penalty = 0.0
        for i in range(4):
            force = np.zeros(6)
            for j in range(self.data.ncon):
                c = self.data.contact[j]
                foot_id = self._foot_geom_ids[i]
                if c.geom1 == foot_id or c.geom2 == foot_id:
                    mujoco.mj_contactForce(self.model, self.data, j, force)
                    f_mag = np.linalg.norm(force[:3])
                    if f_mag > 100.0:
                        penalty += (f_mag - 100.0) ** 2
        return penalty

    # ── main reward ────────────────────────────────────────────────────

    def _get_reward(self, obs, action):
        self._last_action = action.copy()

        reward_fns = dict(
            tracking_lin_vel=self._reward_tracking_lin_vel,
            tracking_ang_vel=self._reward_tracking_ang_vel,
            lin_vel_z=self._reward_lin_vel_z,
            ang_vel_xy=self._reward_ang_vel_xy,
            orientation=self._reward_orientation,
            base_height=self._reward_base_height,
            torques=self._reward_torques,
            dof_vel=self._reward_dof_vel,
            dof_acc=self._reward_dof_acc,
            action_rate=self._reward_action_rate,
            collision=self._reward_collision,
            dof_pos_limits=self._reward_dof_pos_limits,
            feet_air_time=self._reward_feet_air_time,
            stumble=self._reward_stumble,
            stand_still=self._reward_stand_still,
            feet_contact_forces=self._reward_feet_contact_forces,
        )

        total = 0.0
        self._reward_terms = {}
        for name, fn in reward_fns.items():
            val = fn()
            scaled = self._scales[name] * val
            self._reward_terms[name] = scaled
            total += scaled

        # alive bonus
        total += 1.0

        return float(total)

    def _is_terminated(self):
        body_z = self._get_body_z()
        tilt = self._get_body_tilt()

        if body_z < 0.15:
            self._reward_terms["termination"] = self._scales["termination"]
            return True
        if tilt > np.pi / 4:
            self._reward_terms["termination"] = self._scales["termination"]
            return True
        return False

    def _get_info(self):
        return {
            "body_z": self._get_body_z(),
            "body_tilt": self._get_body_tilt(),
            "x_pos": float(self.data.qpos[0]),
            "x_vel": float(self.data.qvel[0]),
            "command": self._command.copy(),
            "reward_terms": dict(self._reward_terms),
        }

    def step(self, action):
        self._step_count += 1
        result = super().step(action)
        self._prev_action = action.copy()
        self._prev_joint_vel = self.data.qvel[6:18].copy()
        return result

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._prev_action[:] = 0
        self._prev_joint_vel[:] = 0
        self._feet_air_time[:] = 0
        self._last_contact[:] = False
        self._reward_terms = {}
        obs, info = super().reset(seed=seed, options=options)
        # Random command after seeding so it's deterministic
        self._command[:] = [
            self.np_random.uniform(0.5, 1.5),
            self.np_random.uniform(-0.3, 0.3),
            self.np_random.uniform(-0.5, 0.5),
        ]
        # Rebuild obs with the new command
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
