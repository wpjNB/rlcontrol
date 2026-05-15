"""Main trot controller: combines phase scheduling, trajectory, IK, and balance."""

import numpy as np
from traditional_control.kinematics import Go2LegKinematics, HOME_QPOS, LEGS
from traditional_control.gait_phase import GaitPhaseScheduler
from traditional_control.foot_trajectory import FootTrajectory

LEG_INDICES = {'FL': 0, 'FR': 3, 'RL': 6, 'RR': 9}
FRONT_LEGS = ['FL', 'FR']
REAR_LEGS = ['RL', 'RR']
LEFT_LEGS = ['FL', 'RL']
RIGHT_LEGS = ['FR', 'RR']


class TrotController:
    """Trot gait controller with IK-based trajectory and balance compensation.

    Args:
        freq: Gait frequency Hz (default 2.0)
        duty_cycle: Stance phase fraction (default 0.6)
        step_height: Foot lift height m (default 0.06)
        target_height: Target body height m (default 0.27)
        tilt_limit: Max tilt before recovery rad (default 0.35)
    """

    def __init__(
        self,
        freq: float = 2.0,
        duty_cycle: float = 0.6,
        step_height: float = 0.06,
        target_height: float = 0.27,
        tilt_limit: float = 0.35,
    ):
        self.kin = Go2LegKinematics()
        self.phase = GaitPhaseScheduler(freq=freq, duty_cycle=duty_cycle)
        self.trajectory = FootTrajectory(step_height=step_height)
        self.target_height = target_height
        self.tilt_limit = tilt_limit

        self.kp_pitch = 1.5
        self.kp_roll = 1.0
        self.kp_height = 5.0

        self._recovering = False
        self._recover_timer = 0.0
        self._recover_duration = 0.5

    def compute(
        self,
        t: float,
        vx: float = 0.0,
        vy: float = 0.0,
        yaw_rate: float = 0.0,
        body_z: float = 0.27,
        pitch: float = 0.0,
        roll: float = 0.0,
        ang_vel_x: float = 0.0,
        ang_vel_y: float = 0.0,
        walking: bool = True,
        dt: float = 0.002,
    ) -> np.ndarray:
        """Compute 12d reference joint angles."""
        tilt = np.sqrt(pitch**2 + roll**2)
        ang_vel_mag = np.sqrt(ang_vel_x**2 + ang_vel_y**2)

        if not self._recovering and (
            tilt > self.tilt_limit or (tilt > 0.2 and ang_vel_mag > 2.0)
        ):
            self._recovering = True
            self._recover_timer = 0.0

        if self._recovering:
            self._recover_timer += dt
            ref = self._recovery_pose(pitch, roll)
            if tilt < 0.1 and self._recover_timer > self._recover_duration:
                self._recovering = False
            return ref

        if not walking:
            return HOME_QPOS.copy()

        phases = self.phase.step(t, dt)
        lateral_offset = self.phase.get_lateral_offset(vy)
        ref = np.zeros(12)

        for leg in LEGS:
            idx = LEG_INDICES[leg]
            p = phases[leg]
            step_len = self.phase.get_step_length(vx, yaw_rate, leg)

            foot_pos = self.trajectory.compute(
                phase_norm=p['phase_norm'],
                is_swing=p['is_swing'],
                step_len=step_len,
                vy_offset=lateral_offset,
            )

            joints = self.kin.solve(leg, foot_pos)
            joints = self.kin.clamp_joints(leg, joints)
            ref[idx:idx + 3] = joints

        # Turning: differential hip angles for left vs right legs
        hip_yaw_offset = yaw_rate * 0.05
        for leg in LEFT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 0] += hip_yaw_offset
        for leg in RIGHT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 0] -= hip_yaw_offset

        ref = self._apply_balance(ref, pitch, roll, body_z, ang_vel_x, ang_vel_y)
        return ref

    def _apply_balance(self, ref, pitch, roll, body_z, ang_vel_x, ang_vel_y):
        """PD-based attitude and height compensation."""
        pitch_corr = self.kp_pitch * pitch + 0.3 * ang_vel_y
        for leg in FRONT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= pitch_corr * 0.3
            ref[idx + 2] += pitch_corr * 0.2
        for leg in REAR_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += pitch_corr * 0.3
            ref[idx + 2] -= pitch_corr * 0.2

        roll_corr = self.kp_roll * roll + 0.2 * ang_vel_x
        for leg in RIGHT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= roll_corr * 0.2
            ref[idx + 0] -= roll_corr * 0.1
        for leg in LEFT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += roll_corr * 0.2
            ref[idx + 0] += roll_corr * 0.1

        height_err = self.target_height - body_z
        height_corr = self.kp_height * height_err
        for leg in LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= height_corr * 0.2
            ref[idx + 2] += height_corr * 0.15

        return ref

    def _recovery_pose(self, pitch, roll):
        """Crouch pose to recover from near-fall."""
        ref = HOME_QPOS.copy()
        for leg in LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 2] -= 0.3

        pitch_corr = 0.8 * pitch
        for leg in FRONT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= pitch_corr * 0.4
        for leg in REAR_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += pitch_corr * 0.4

        roll_corr = 0.6 * roll
        for leg in RIGHT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= roll_corr * 0.3
        for leg in LEFT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += roll_corr * 0.3

        return ref

    @property
    def is_recovering(self) -> bool:
        return self._recovering
