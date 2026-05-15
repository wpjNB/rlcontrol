"""PD joint controller for MuJoCo quadruped robots."""

import numpy as np


class PDController:
    """PD position controller for joint-space tracking.

    Computes torque: τ = kp * (target - q) - kd * qvel

    Args:
        n_joints: Number of controlled joints
        kp: Position gain (scalar or array)
        kd: Velocity gain (scalar or array)
    """

    def __init__(self, n_joints: int = 12, kp: float = 50.0, kd: float = 1.5):
        self.n_joints = n_joints
        self.kp = np.full(n_joints, kp) if np.isscalar(kp) else np.asarray(kp)
        self.kd = np.full(n_joints, kd) if np.isscalar(kd) else np.asarray(kd)

    def compute(self, target: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Compute joint torques.

        Args:
            target: Target joint angles (n_joints,)
            qpos: Current joint positions (n_joints,)
            qvel: Current joint velocities (n_joints,)

        Returns:
            Torque array (n_joints,)
        """
        return self.kp * (target - qpos) - self.kd * qvel

    def apply(self, target: np.ndarray, data) -> None:
        """Compute and apply torques directly to MuJoCo data.

        Args:
            target: Target joint angles (n_joints,)
            data: MuJoCo MjData (reads qpos[7:7+n], qvel[6:6+n], writes ctrl[:n])
        """
        qpos = data.qpos[7:7 + self.n_joints]
        qvel = data.qvel[6:6 + self.n_joints]
        data.ctrl[:self.n_joints] = self.compute(target, qpos, qvel)
