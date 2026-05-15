"""Trot gait phase scheduler with step length calculation."""

import numpy as np

TROT_OFFSETS = {
    'FL': np.pi,
    'FR': 0.0,
    'RL': np.pi,
    'RR': 0.0,
}

LEGS = ['FL', 'FR', 'RL', 'RR']
LEFT_LEGS = {'FL', 'RL'}
RIGHT_LEGS = {'FR', 'RR'}


class GaitPhaseScheduler:
    """Manages swing/stance phase timing for each leg in a trot gait.

    Args:
        freq: Gait frequency in Hz (default 2.0)
        duty_cycle: Fraction of cycle spent in stance (default 0.6)
    """

    def __init__(self, freq: float = 2.0, duty_cycle: float = 0.6):
        self.freq = freq
        self.duty_cycle = duty_cycle
        self.swing_end = 2 * np.pi * (1 - duty_cycle)

    def step(self, t: float, dt: float) -> dict:
        """Compute phase info for all legs at time t.

        Returns:
            Dict mapping leg name -> {
                'phase': raw phase 0~2pi,
                'is_swing': True if in swing phase,
                'phase_norm': normalized progress within current phase (0~1)
            }
        """
        omega = 2 * np.pi * self.freq
        result = {}

        for leg in LEGS:
            phase = (omega * t + TROT_OFFSETS[leg]) % (2 * np.pi)
            is_swing = phase < self.swing_end

            if is_swing:
                phase_norm = phase / self.swing_end
            else:
                phase_norm = (phase - self.swing_end) / (2 * np.pi - self.swing_end)

            result[leg] = {
                'phase': phase,
                'is_swing': is_swing,
                'phase_norm': phase_norm,
            }

        return result

    def get_step_length(self, vx: float, yaw_rate: float, leg: str) -> float:
        """Calculate step length for a leg based on velocity commands."""
        base_step = vx / max(self.freq, 0.1)
        turn_offset = yaw_rate * 0.1 / max(self.freq, 0.1)

        if leg in LEFT_LEGS:
            return base_step + turn_offset
        else:
            return base_step - turn_offset

    def get_lateral_offset(self, vy: float) -> float:
        """Calculate lateral foot offset from vy command."""
        return vy / max(self.freq, 0.1)
