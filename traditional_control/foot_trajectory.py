"""Cartesian foot trajectory generation for trot gait."""

import numpy as np


class FootTrajectory:
    """Generates foot trajectories in Cartesian space.

    Swing phase: sinusoidal lift + linear forward sweep.
    Stance phase: foot planted, body moves forward over it.

    Args:
        step_height: Maximum foot lift height in meters (default 0.06)
        step_length_max: Maximum step length in meters (default 0.2)
    """

    def __init__(self, step_height: float = 0.06, step_length_max: float = 0.2):
        self.step_height = step_height
        self.step_length_max = step_length_max

    def compute(
        self,
        phase_norm: float,
        is_swing: bool,
        step_len: float,
        vy_offset: float,
    ) -> np.ndarray:
        """Compute foot target position relative to hip joint.

        Args:
            phase_norm: Normalized progress within current phase (0~1)
            is_swing: True if in swing phase
            step_len: Desired step length in meters (sign = direction)
            vy_offset: Lateral offset in meters

        Returns:
            [x, y, z] foot position relative to hip
        """
        step_len = np.clip(step_len, -self.step_length_max, self.step_length_max)

        if is_swing:
            x = -step_len / 2 + step_len * phase_norm
            z = self.step_height * np.sin(np.pi * phase_norm)
        else:
            x = step_len / 2 - step_len * phase_norm
            z = 0.0

        y = vy_offset

        return np.array([x, y, z])
