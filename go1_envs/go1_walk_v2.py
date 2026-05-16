"""Go1WalkV2Env: Go1 walking with full reward suite from IsaacGym/Legged Gym."""

import numpy as np
import mujoco
from go1_envs.base import Go1BaseEnv, HOME_QPOS
from go1_envs.gait import TrotGait

SCENE_FILE = "go1_envs/scenes/flat_scene.xml"

# Model structure
FOOT_BODY_NAMES = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
NON_FOOT_BODY_NAMES = [
    "trunk", "FR_hip", "FR_thigh", "FL_hip", "FL_thigh",
    "RR_hip", "RR_thigh", "RL_hip", "RL_thigh",
]
TARGET_BASE_HEIGHT = 0.27


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
        difficulty=1.0,
        # reward scales
        tracking_lin_vel_scale=2.0,
        tracking_ang_vel_scale=1.0,
        lin_vel_z_scale=-1.0,
        ang_vel_xy_scale=-0.02,
        orientation_scale=-0.5,
        base_height_scale=-10.0,
        torques_scale=-0.0001,
        dof_vel_scale=-0.0002,
        dof_acc_scale=-1e-7,
        action_rate_scale=-0.005,
        collision_scale=-0.5,
        termination_scale=-2.0,
        dof_pos_limits_scale=-1.0,
        feet_air_time_scale=5.0,
        stumble_scale=-2.0,
        stand_still_scale=-0.5,
        feet_contact_forces_scale=-0.001,
        contact_consistency_scale=2.0,
        ref_tracking_scale=5.0,
        # tracking sigma
        tracking_sigma=0.25,
        **kwargs,
    ):
        super().__init__(xml_file=SCENE_FILE, render_mode=render_mode, **kwargs)

        # Curriculum: 0.0 = stand still with light penalties, 1.0 = full task
        self._difficulty = float(difficulty)

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

        # Gait phase (trot: diagonal pairs, FR/RL in phase, FL/RR offset by 0.5)
        self._gait_period = 0.5  # seconds
        # foot order: FR, FL, RR, RL  (matches FOOT_BODY_NAMES)
        self._foot_phase_offsets = np.array([0.0, 0.5, 0.5, 0.0])

        # Reference gait generator
        self._gait_gen = TrotGait(freq=3.0, thigh_amp=0.25, calf_amp=0.2)

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
            contact_consistency=contact_consistency_scale,
            ref_tracking=ref_tracking_scale,
        )
        self._tracking_sigma = tracking_sigma

        # Override obs space: 39 + 3 (commanded_vel) + 2 (gait phase) + 12 (ref joints)
        from gymnasium import spaces
        obs_dim = 56
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._step_count = 0

    def _get_obs(self):
        base_obs = super()._get_obs()  # 39d
        cmd = self._command.astype(np.float32)
        phase = self._gait_phase()
        phase_enc = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32)
        ref_joints = self._get_ref_joint_pos().astype(np.float32)  # 12d
        return np.concatenate([base_obs, cmd, phase_enc, ref_joints])

    def set_command(self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0):
        self._command[:] = [vx, vy, yaw_rate]

    def set_difficulty(self, d: float):
        self._difficulty = float(np.clip(d, 0.0, 1.0))

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

    # ── gait phase ─────────────────────────────────────────────────────

    def _gait_phase(self):
        """Current gait phase in [0, 1)."""
        dt = self.frame_skip * self.model.opt.timestep
        t = self._step_count * dt
        return (t % self._gait_period) / self._gait_period

    def _get_ref_joint_pos(self):
        """Generate reference joint positions using TrotGait generator."""
        dt = self.frame_skip * self.model.opt.timestep
        t = self._step_count * dt
        vx = float(self._command[0])
        vy = float(self._command[1])
        yaw = float(self._command[2])
        return self._gait_gen.step(t, vx=vx, vy=vy, yaw_rate=yaw)

    def _reward_contact_consistency(self):
        """Reward feet contact matching expected trot gait pattern."""
        phase = self._gait_phase()
        contact = self._get_foot_contacts()
        reward = 0.0
        for i in range(4):
            foot_phase = (phase + self._foot_phase_offsets[i]) % 1.0
            is_stance = foot_phase < 0.55
            is_contact = contact[i] > 0.5
            if is_stance == is_contact:
                reward += 1.0
        return reward / 4.0  # normalize to [0, 1]

    def _reward_ref_tracking(self):
        """Reward tracking the reference gait joint positions."""
        joint_pos = self.data.qpos[7:19]
        ref = self._get_ref_joint_pos()
        err = np.sum((joint_pos - ref) ** 2)
        return float(np.exp(-err / 0.5))

    # ── main reward ────────────────────────────────────────────────────

    # Penalty terms that get scaled by difficulty
    _PENALTY_TERMS = frozenset([
        "lin_vel_z", "ang_vel_xy", "orientation", "base_height",
        "torques", "dof_vel", "dof_acc", "action_rate",
        "collision", "dof_pos_limits", "stumble", "stand_still",
        "feet_contact_forces",
    ])

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
            contact_consistency=self._reward_contact_consistency,
            stumble=self._reward_stumble,
            stand_still=self._reward_stand_still,
            feet_contact_forces=self._reward_feet_contact_forces,
            ref_tracking=self._reward_ref_tracking,
        )

        total = 0.0
        self._reward_terms = {}
        d = self._difficulty
        for name, fn in reward_fns.items():
            val = fn()
            scale = self._scales[name]
            # Scale penalty terms by difficulty; positive rewards stay full
            if name in self._PENALTY_TERMS and scale < 0:
                scale *= d
            scaled = scale * val
            self._reward_terms[name] = scaled
            total += scaled

        # alive bonus (scale up with difficulty to compensate for more penalties)
        total += 1.0 + 3.0 * d

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
        # Random command scaled by difficulty
        d = self._difficulty
        self._command[:] = [
            self.np_random.uniform(0.5 * d, 1.5 * d),
            self.np_random.uniform(-0.3 * d, 0.3 * d),
            self.np_random.uniform(-0.5 * d, 0.5 * d),
        ]
        # Rebuild obs with the new command
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
