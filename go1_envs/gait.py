"""Trot gait generator for Go1 quadruped.

Generates reference joint angles for trot gait at a given speed.
Joint layout: [hip, thigh, knee] × [FR, FL, RR, RL]

Usage:
    gen = TrotGait(freq=3.0)
    ref = gen.step(t=0.0, vx=0.0)            # in-place stepping
    ref = gen.step(t=0.0, vx=1.0)            # forward 1 m/s
    ref = gen.step(t=0.0, vy=0.5)            # lateral left 0.5 m/s
    ref = gen.step(t=0.0, vx=1.0, yaw_rate=0.5)  # forward + turn
"""

import numpy as np

# Home joint angles (standing pose)
HOME_QPOS = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])

# Trot phase offsets: FR/RL in phase, FL/RR offset by π
TROT_PHASES = [0.0, np.pi, np.pi, 0.0]

# Leg indices
HIP, THIGH, KNEE = 0, 1, 2


class TrotGait:
    """Generates trot gait reference joint trajectories.

    Args:
        freq: Gait frequency in Hz (default 3.0)
        thigh_amp: Thigh oscillation amplitude in rad (default 0.25)
        calf_amp: Calf oscillation amplitude in rad (default 0.2)
        hip_amp: Hip oscillation amplitude for turning in rad (default 0.15)
    """

    def __init__(
        self,
        freq: float = 3.0,
        thigh_amp: float = 0.25,
        calf_amp: float = 0.2,
        hip_amp: float = 0.15,
    ):
        self.freq = freq
        self.thigh_amp = thigh_amp
        self.calf_amp = calf_amp
        self.hip_amp = hip_amp

    def step(self, t: float, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0) -> np.ndarray:
        """Generate reference joint angles at time t.

        Args:
            t: Current time in seconds
            vx: Forward velocity in m/s (0 = in-place stepping)
            vy: Lateral velocity in m/s (positive = left)
            yaw_rate: Yaw rate in rad/s (0 = straight)

        Returns:
            12d numpy array of target joint angles
        """
        omega = 2.0 * np.pi * self.freq
        ref = np.zeros(12)

        # Velocity → thigh offset (forward lean bias)
        fwd_offset = vx * 0.15

        for leg in range(4):
            phase = TROT_PHASES[leg]
            base = leg * 3

            # Hip: yaw + lateral
            #   yaw:  right legs positive, left legs negative
            #   lateral: right legs negative, left legs positive (push body left)
            hip = yaw_rate * self.hip_amp
            if leg in [0, 2]:  # right legs (FR, RR)
                ref[base + HIP] = hip - vy * 0.1
            else:              # left legs (FL, RL)
                ref[base + HIP] = -hip + vy * 0.1

            # Thigh: oscillation + forward bias
            ref[base + THIGH] = (
                HOME_QPOS[base + THIGH]
                + fwd_offset
                + self.thigh_amp * np.sin(omega * t + phase)
            )

            # Calf: cos (90° offset from thigh) for natural leg motion
            ref[base + KNEE] = (
                HOME_QPOS[base + KNEE]
                + self.calf_amp * np.cos(omega * t + phase)
            )

        return ref

    def __repr__(self):
        return f"TrotGait(freq={self.freq}, thigh={self.thigh_amp}, calf={self.calf_amp})"
