"""Trot 步态相位调度器 — 管理每条腿的摆动/支撑相切换和步长计算。

Trot（对角步态）是最常见的四足步态：
- 对角腿同步运动：FR+RL 同相，FL+RR 同相，两组相差 π
- 每条腿在一个周期内交替：摆动相（抬腿前摆）和支撑相（踩地后推）

本模块负责：
- 根据时间 t 计算每条腿的当前相位
- 判断每条腿处于摆动相还是支撑相
- 根据速度命令 (vx, vy, yaw_rate) 计算每条腿的步长
"""

import numpy as np

# Trot 步态的相位偏移
# FR 和 RL 同相（0），FL 和 RR 同相（π），形成对角对称
TROT_OFFSETS = {
    'FL': np.pi,   # 左前腿
    'FR': 0.0,     # 右前腿
    'RL': np.pi,   # 左后腿
    'RR': 0.0,     # 右后腿
}

LEGS = ['FL', 'FR', 'RL', 'RR']
LEFT_LEGS = {'FL', 'RL'}    # 左侧腿
RIGHT_LEGS = {'FR', 'RR'}   # 右侧腿


class GaitPhaseScheduler:
    """Trot 步态相位调度器。

    Args:
        freq: 步频 Hz（默认 2.0，即每秒 2 步）
        duty_cycle: 支撑相比例（默认 0.6，即 60% 时间脚踩在地上）
    """

    def __init__(self, freq: float = 2.0, duty_cycle: float = 0.6):
        self.freq = freq
        self.duty_cycle = duty_cycle
        # 摆动相结束的相位角 = 2π × (1 - 支撑相比例)
        # 例如 duty=0.6 时，摆动相占 0~0.8π，支撑相占 0.8π~2π
        self.swing_end = 2 * np.pi * (1 - duty_cycle)

    def step(self, t: float, dt: float) -> dict:
        """计算时刻 t 所有腿的相位信息。

        Args:
            t: 当前时间（秒）
            dt: 时间步长（秒，本函数未使用，保留接口一致性）

        Returns:
            字典，key=腿名，value={
                'phase': 原始相位 0~2π,
                'is_swing': 是否在摆动相,
                'phase_norm': 在当前阶段内的归一化进度 0~1
            }
        """
        omega = 2 * np.pi * self.freq  # 角频率
        result = {}

        for leg in LEGS:
            # 计算当前相位：ωt + 初始偏移，取模到 [0, 2π)
            phase = (omega * t + TROT_OFFSETS[leg]) % (2 * np.pi)

            # 判断摆动/支撑：相位 < swing_end 为摆动相
            is_swing = phase < self.swing_end

            # 归一化进度：在当前阶段内走了多远（0=刚进入，1=即将切换）
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
        """根据速度命令计算某条腿的步长。

        步长 = 基础步长 + 转向差速偏移
        - 基础步长 = vx / freq（速度越快步子越大）
        - 转向偏移 = yaw_rate * 0.1 / freq（左转时左腿步子大、右腿步子小）

        Args:
            vx: 前进速度 (m/s)
            yaw_rate: 偏航角速度 (rad/s, 正=左转)
            leg: 腿名

        Returns:
            步长 (m)，正=向前，负=向后
        """
        base_step = vx / max(self.freq, 0.1)
        turn_offset = yaw_rate * 0.1 / max(self.freq, 0.1)

        if leg in LEFT_LEGS:
            return base_step + turn_offset   # 左腿：基础 + 偏移
        else:
            return base_step - turn_offset   # 右腿：基础 - 偏移

    def get_lateral_offset(self, vy: float) -> float:
        """根据侧移速度计算足端横向偏移。"""
        return vy / max(self.freq, 0.1)
