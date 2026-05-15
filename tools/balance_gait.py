"""Feedback-aware trot gait controller with balance compensation.

Layers:
  1. Base gait: sinusoidal trot trajectory (from TrotGait)
  2. Attitude compensation: adjust legs when body tilts/rolls
  3. Height control: adjust leg extension to maintain target height
  4. Fall recovery: detect imminent fall, trigger recovery pose

Joint layout: [hip, thigh, knee] × [FR, FL, RR, RL]
"""

import numpy as np
from go1_envs.gait import TrotGait, HOME_QPOS

HIP, THIGH, KNEE = 0, 1, 2
RIGHT_LEGS = [0, 2]  # FR, RR
LEFT_LEGS = [1, 3]   # FL, RL
FRONT_LEGS = [0, 1]  # FR, FL
REAR_LEGS = [2, 3]   # RR, RL


class BalanceGait:
    """Trot gait with balance feedback.

    Args:
        freq: Gait frequency Hz
        target_height: Target body height m
        tilt_limit: Max tilt before recovery rad (~20°)
    """

    def __init__(
        self,
        freq: float = 3.0,
        target_height: float = 0.27,
        tilt_limit: float = 0.35,
    ):
        self.gait = TrotGait(freq=freq)
        self.target_height = target_height
        self.tilt_limit = tilt_limit

        # Attitude compensation gains
        self.kp_pitch = 1.5   # pitch → front/rear thigh offset
        self.kp_roll = 1.0    # roll → left/right hip offset
        self.kp_height = 5.0  # height error → all thigh offset

        # Fall recovery state
        self._recovering = False
        self._recover_timer = 0.0
        self._recover_duration = 0.5  # seconds in recovery pose

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
        """Compute reference joint angles with balance feedback.

        Args:
            t: Current time s
            vx, vy, yaw_rate: Velocity commands
            body_z: Body height m
            pitch: Body pitch rad (positive = nose up)
            roll: Body roll rad (positive = tilt right)
            ang_vel_x, ang_vel_y: Angular velocity rad/s
            walking: Whether gait is active
            dt: Timestep for recovery timer

        Returns:
            12d target joint angles
        """
        # --- Fall detection ---
        tilt = np.sqrt(pitch**2 + roll**2)
        ang_vel_mag = np.sqrt(ang_vel_x**2 + ang_vel_y**2)

        # Trigger recovery if tilt too large or angular velocity too high while tilted
        if not self._recovering and (tilt > self.tilt_limit or (tilt > 0.2 and ang_vel_mag > 2.0)):
            self._recovering = True
            self._recover_timer = 0.0

        # --- Recovery mode ---
        if self._recovering:
            self._recover_timer += dt
            ref = self._recovery_pose(pitch, roll)
            # Exit recovery when body is back to stable
            if tilt < 0.1 and self._recover_timer > self._recover_duration:
                self._recovering = False
            return ref

        # --- Normal mode ---
        if walking:
            ref = self.gait.step(t, vx=vx, vy=vy, yaw_rate=yaw_rate)
        else:
            ref = HOME_QPOS.copy()

        # --- Attitude compensation (applies to both walking and standing) ---
        ref = self._apply_balance(ref, pitch, roll, body_z, ang_vel_x, ang_vel_y)

        return ref

    def _apply_balance(
        self,
        ref: np.ndarray,
        pitch: float,
        roll: float,
        body_z: float,
        ang_vel_x: float,
        ang_vel_y: float,
    ) -> np.ndarray:
        """Adjust reference joints to compensate for body tilt and height error."""
        # Pitch compensation: tilt forward → extend front legs, retract rear
        # Use PD: proportional on pitch, derivative on ang_vel_y (pitch rate)
        pitch_corr = self.kp_pitch * pitch + 0.3 * ang_vel_y

        for leg in FRONT_LEGS:
            ref[leg * 3 + THIGH] -= pitch_corr * 0.3  # extend front
            ref[leg * 3 + KNEE] += pitch_corr * 0.2
        for leg in REAR_LEGS:
            ref[leg * 3 + THIGH] += pitch_corr * 0.3  # retract rear
            ref[leg * 3 + KNEE] -= pitch_corr * 0.2

        # Roll compensation: tilt right → extend right legs, retract left
        roll_corr = self.kp_roll * roll + 0.2 * ang_vel_x

        for leg in RIGHT_LEGS:
            ref[leg * 3 + THIGH] -= roll_corr * 0.2
            ref[leg * 3 + HIP] -= roll_corr * 0.1
        for leg in LEFT_LEGS:
            ref[leg * 3 + THIGH] += roll_corr * 0.2
            ref[leg * 3 + HIP] += roll_corr * 0.1

        # Height compensation: too low → extend all legs
        height_err = self.target_height - body_z
        height_corr = self.kp_height * height_err
        for leg in range(4):
            ref[leg * 3 + THIGH] -= height_corr * 0.2
            ref[leg * 3 + KNEE] += height_corr * 0.15

        return ref

    def _recovery_pose(self, pitch: float, roll: float) -> np.ndarray:
        """Crouch pose to recover from near-fall.

        Strategy: lower center of mass by crouching, counter-tilt to re-balance.
        """
        ref = HOME_QPOS.copy()

        # Crouch: extend knees more, lower body
        for leg in range(4):
            ref[leg * 3 + KNEE] = HOME_QPOS[leg * 3 + KNEE] + 0.3  # more flex

        # Counter-tilt
        pitch_corr = 0.8 * pitch
        for leg in FRONT_LEGS:
            ref[leg * 3 + THIGH] -= pitch_corr * 0.4
        for leg in REAR_LEGS:
            ref[leg * 3 + THIGH] += pitch_corr * 0.4

        roll_corr = 0.6 * roll
        for leg in RIGHT_LEGS:
            ref[leg * 3 + THIGH] -= roll_corr * 0.3
        for leg in LEFT_LEGS:
            ref[leg * 3 + THIGH] += roll_corr * 0.3

        return ref

    @property
    def is_recovering(self) -> bool:
        return self._recovering
