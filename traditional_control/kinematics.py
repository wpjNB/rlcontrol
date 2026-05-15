"""Go2 四足机器人腿部运动学 — 正运动学(FK)、逆运动学(IK)、关节限位。

本模块是整个控制系统的最底层，负责：
- 存储 Go2 的运动学参数（腿长、关节限位等）
- 正运动学：已知关节角度 → 计算足端位置
- 逆运动学：已知足端目标位置 → 计算关节角度
- 关节限位裁剪：确保关节角不超出硬件限制
"""

import numpy as np

# ============================================================
# Go2 运动学常量（从 MuJoCo 模型 unitree_go2/scene.xml 提取）
# ============================================================

# 髋关节相对于基座中心的偏移量 [x前后, y左右, z上下]
HIP_OFFSET = {
    'FL': np.array([0.1934,  0.0465, 0.0]),   # 左前腿
    'FR': np.array([0.1934, -0.0465, 0.0]),   # 右前腿
    'RL': np.array([-0.1934,  0.0465, 0.0]),  # 左后腿
    'RR': np.array([-0.1934, -0.0465, 0.0]),  # 右后腿
}

HIP_LINK = 0.0955   # 髋外展连杆长度 (y方向, hip→thigh)
L1 = 0.213          # 大腿长度 (z方向, thigh→calf)
L2 = 0.213          # 小腿长度 (z方向, calf→foot)

# 关节限位 [最小值, 最大值]（弧度）
# 前腿和后腿的大腿限位不同，其他相同
JOINT_LIMITS = {
    'FL': [(-1.047, 1.047), (-1.571, 3.491), (-2.723, -0.838)],  # [髋, 大腿, 小腿]
    'FR': [(-1.047, 1.047), (-1.571, 3.491), (-2.723, -0.838)],
    'RL': [(-1.047, 1.047), (-0.524, 4.538), (-2.723, -0.838)],  # 后腿大腿范围更大
    'RR': [(-1.047, 1.047), (-0.524, 4.538), (-2.723, -0.838)],
}

# 站立姿态的关节角度 [髋, 大腿, 小腿] × 4条腿
# 大腿 0.9rad ≈ 向前倾斜51°，小腿 -1.8rad ≈ 弯曲103°
HOME_QPOS = np.array([
    0.0, 0.9, -1.8,   # 左前腿 FL
    0.0, 0.9, -1.8,   # 右前腿 FR
    0.0, 0.9, -1.8,   # 左后腿 RL
    0.0, 0.9, -1.8,   # 右后腿 RR
])

LEGS = ['FL', 'FR', 'RL', 'RR']


class Go2LegKinematics:
    """Go2 单腿 3-DOF 运动学求解器。

    每条腿有 3 个关节：
    - hip（髋关节）：绕 x 轴旋转，控制腿的外展/内收
    - thigh（大腿关节）：绕 y 轴旋转，控制大腿前后摆动
    - calf（小腿关节）：绕 y 轴旋转，控制小腿弯曲/伸展

    坐标系：足端位置相对于髋关节，x=前, y=左, z=上
    """

    def __init__(self):
        self.L1 = L1
        self.L2 = L2
        self.hip_link = HIP_LINK

    def forward(self, leg: str, joints: np.ndarray) -> np.ndarray:
        """正运动学：关节角度 → 足端位置（相对于髋关节）。

        计算步骤：
        1. 髋关节旋转产生 y 方向偏移
        2. 大腿旋转产生 x, z 方向偏移
        3. 小腿旋转产生额外 x, z 偏移
        4. 累加得到最终足端位置

        Args:
            leg: 腿名 'FL', 'FR', 'RL', 'RR'
            joints: [髋角, 大腿角, 小腿角] 弧度

        Returns:
            [x, y, z] 足端位置（相对于髋关节）
        """
        q_hip, q_thigh, q_calf = joints

        # 髋外展：y = hip_link * cos(q_hip) - hip_link
        # q_hip=0 时 y=0（腿垂直向下），q_hip>0 时 y<0（向内收）
        y = self.hip_link * np.cos(q_hip) - self.hip_link

        # 大腿在 x-z 平面内的 FK
        x1 = self.L1 * np.sin(q_thigh)
        z1 = -self.L1 * np.cos(q_thigh)

        # 小腿相对于大腿的 FK
        x2 = self.L2 * np.sin(q_thigh + q_calf)
        z2 = -self.L2 * np.cos(q_thigh + q_calf)

        # 累加得到总位移
        x = x1 + x2
        z = z1 + z2

        # 髋旋转对 x 分量的影响
        x_eff = x * np.cos(q_hip)

        return np.array([x_eff, y, z])

    def solve(self, leg: str, foot_pos: np.ndarray) -> np.ndarray:
        """逆运动学：足端目标位置 → 关节角度。

        求解步骤：
        1. 从 y 坐标求髋外展角
        2. 将 x 投影到大腿-小腿平面
        3. 用余弦定理求小腿角
        4. 用几何关系求大腿角

        Args:
            leg: 腿名 'FL', 'FR', 'RL', 'RR'
            foot_pos: [x, y, z] 足端目标位置（相对于髋关节）

        Returns:
            [髋角, 大腿角, 小腿角] 弧度
        """
        x, y, z = foot_pos

        # --- 求髋外展角 ---
        # 从 y = hip_link * cos(q_hip) - hip_link 反推 q_hip
        cos_hip = np.clip((y + self.hip_link) / self.hip_link, -1.0, 1.0)
        q_hip = np.arccos(cos_hip)
        if y < 0:
            q_hip = -q_hip

        # --- 投影到大腿-小腿平面 ---
        # 髋旋转后，有效 x 坐标会缩小
        x_eff = x / max(np.cos(q_hip), 0.1)

        # --- 2 连杆 IK 求解 ---
        d_sq = x_eff**2 + z**2
        d = np.sqrt(d_sq)

        # 余弦定理求小腿角
        # cos(calf) = (d² - L1² - L2²) / (2*L1*L2)
        cos_calf = (d_sq - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_calf = np.clip(cos_calf, -1.0, 1.0)
        q_calf = -np.arccos(cos_calf)  # 负值表示小腿弯曲

        # 大腿角 = 足端方位角 - 小腿偏移角
        alpha = np.arctan2(x_eff, -z)
        beta = np.arctan2(self.L2 * np.sin(q_calf),
                          self.L1 + self.L2 * np.cos(q_calf))
        q_thigh = alpha - beta

        return np.array([q_hip, q_thigh, q_calf])

    def clamp_joints(self, leg: str, joints: np.ndarray) -> np.ndarray:
        """将关节角度裁剪到硬件允许范围内。"""
        limits = JOINT_LIMITS[leg]
        result = np.copy(joints)
        for i in range(3):
            result[i] = np.clip(result[i], limits[i][0], limits[i][1])
        return result

    def get_limits(self, leg: str):
        """返回指定腿的关节限位。"""
        return JOINT_LIMITS[leg]
