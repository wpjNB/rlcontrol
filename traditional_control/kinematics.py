"""Go2 quadruped leg kinematics — FK, IK, joint limits."""

import numpy as np

# Go2 kinematic parameters (extracted from MuJoCo model)
HIP_OFFSET = {
    'FL': np.array([0.1934,  0.0465, 0.0]),
    'FR': np.array([0.1934, -0.0465, 0.0]),
    'RL': np.array([-0.1934,  0.0465, 0.0]),
    'RR': np.array([-0.1934, -0.0465, 0.0]),
}

HIP_LINK = 0.0955   # hip abduction link length (y direction)
L1 = 0.213          # thigh length (z direction)
L2 = 0.213          # calf length (z direction)

# Joint limits from MuJoCo model
JOINT_LIMITS = {
    'FL': [(-1.047, 1.047), (-1.571, 3.491), (-2.723, -0.838)],
    'FR': [(-1.047, 1.047), (-1.571, 3.491), (-2.723, -0.838)],
    'RL': [(-1.047, 1.047), (-0.524, 4.538), (-2.723, -0.838)],
    'RR': [(-1.047, 1.047), (-0.524, 4.538), (-2.723, -0.838)],
}

# Standing pose joint angles
HOME_QPOS = np.array([
    0.0, 0.9, -1.8,   # FL: hip, thigh, knee
    0.0, 0.9, -1.8,   # FR
    0.0, 0.9, -1.8,   # RL
    0.0, 0.9, -1.8,   # RR
])

LEGS = ['FL', 'FR', 'RL', 'RR']


class Go2LegKinematics:
    """3-DOF leg kinematics for Unitree Go2."""

    def __init__(self):
        self.L1 = L1
        self.L2 = L2
        self.hip_link = HIP_LINK

    def forward(self, leg: str, joints: np.ndarray) -> np.ndarray:
        """Forward kinematics: joint angles -> foot position relative to hip.

        Args:
            leg: 'FL', 'FR', 'RL', or 'RR'
            joints: [hip, thigh, calf] angles in radians

        Returns:
            [x, y, z] foot position relative to hip joint
        """
        q_hip, q_thigh, q_calf = joints

        # Hip abduction: y offset
        y = self.hip_link * np.cos(q_hip) - self.hip_link

        # Planar FK (thigh + calf in the x-z plane)
        x1 = self.L1 * np.sin(q_thigh)
        z1 = -self.L1 * np.cos(q_thigh)

        x2 = self.L2 * np.sin(q_thigh + q_calf)
        z2 = -self.L2 * np.cos(q_thigh + q_calf)

        x = x1 + x2
        z = z1 + z2

        # Apply hip rotation to x component
        x_eff = x * np.cos(q_hip)

        return np.array([x_eff, y, z])

    def solve(self, leg: str, foot_pos: np.ndarray) -> np.ndarray:
        """Inverse kinematics: foot position -> joint angles.

        Args:
            leg: 'FL', 'FR', 'RL', or 'RR'
            foot_pos: [x, y, z] foot position relative to hip joint

        Returns:
            [hip, thigh, calf] joint angles in radians
        """
        x, y, z = foot_pos

        # Hip abduction angle
        cos_hip = np.clip((y + self.hip_link) / self.hip_link, -1.0, 1.0)
        q_hip = np.arccos(cos_hip)
        if y < 0:
            q_hip = -q_hip

        # Project into the sagittal plane
        x_eff = x / max(np.cos(q_hip), 0.1)

        # 2-link IK in the sagittal plane (x_eff, z)
        d_sq = x_eff**2 + z**2
        d = np.sqrt(d_sq)

        # Cosine rule for calf angle
        cos_calf = (d_sq - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_calf = np.clip(cos_calf, -1.0, 1.0)
        q_calf = -np.arccos(cos_calf)  # negative = bent

        # Thigh angle
        alpha = np.arctan2(x_eff, -z)
        beta = np.arctan2(self.L2 * np.sin(q_calf),
                          self.L1 + self.L2 * np.cos(q_calf))
        q_thigh = alpha - beta

        return np.array([q_hip, q_thigh, q_calf])

    def clamp_joints(self, leg: str, joints: np.ndarray) -> np.ndarray:
        """Clamp joint angles to hardware limits."""
        limits = JOINT_LIMITS[leg]
        result = np.copy(joints)
        for i in range(3):
            result[i] = np.clip(result[i], limits[i][0], limits[i][1])
        return result

    def get_limits(self, leg: str):
        """Return joint limits for a leg."""
        return JOINT_LIMITS[leg]
