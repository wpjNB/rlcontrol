"""笛卡尔空间足端轨迹生成器 — 规划每条腿在摆动/支撑相的足端运动轨迹。

本模块将步态相位转化为具体的足端位置目标：
- 摆动相（脚离地）：五次多项式前摆（起止速度/加速度为零）+ 非对称抬腿
- 支撑相（脚踩地）：足端固定不动，身体从脚上方移过

输出的足端位置 [x, y, z] 是相对于髋关节的，后续由 IK 模块转为关节角。
"""

import numpy as np


def _quintic(t: float) -> float:
    """五次多项式插值 s(t)，满足 s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0。

    保证起止速度和加速度均为零，消除摆动→支撑切换时的冲击。
    """
    return t * t * t * (6 * t * t - 15 * t + 10)


def _asym_z(t: float, peak_ratio: float = 0.4) -> float:
    """非对称抬腿曲线：前半段快速抬升，后半段缓慢下降。

    Args:
        t: 归一化时间 0~1
        peak_ratio: 最高点位置比例 (0~1)，越小越早到达最高点

    Returns:
        0~1 的高度曲线
    """
    if t <= peak_ratio:
        # 前半段：正弦快速抬升到 1
        return np.sin(np.pi * t / (2 * peak_ratio))
    else:
        # 后半段：余弦缓慢下降到 0
        return np.cos(np.pi * (t - peak_ratio) / (2 * (1 - peak_ratio)))


class FootTrajectory:
    """笛卡尔空间足端轨迹生成器。

    Args:
        step_height: 最大抬腿高度 (m)，默认 0.08
        step_length_max: 最大步长 (m)，用于裁剪，默认 0.3
        standing_height: 站立时足端到髋关节的垂直距离 (m)，默认 0.265
        z_peak_ratio: 抬腿最高点的相位比例，默认 0.4（偏前，快速抬起后缓落）
    """

    def __init__(self, step_height: float = 0.08, step_length_max: float = 0.3,
                 standing_height: float = 0.265, z_peak_ratio: float = 0.4):
        self.step_height = step_height
        self.step_length_max = step_length_max
        self.standing_height = standing_height
        self.z_peak_ratio = z_peak_ratio

    def compute(
        self,
        phase_norm: float,
        is_swing: bool,
        step_len: float,
        lateral_step: float,
    ) -> np.ndarray:
        """计算足端目标位置（相对于髋关节）。

        摆动相轨迹：
          x: 五次多项式 s(t)，起止速度/加速度为零
          y: 五次多项式，同上
          z: 非对称曲线，前快抬后慢落

        支撑相轨迹：
          x = +step_len/2 → -step_len/2  （足端固定，身体前移）
          y = +lateral_step/2 → -lateral_step/2  （侧移）
          z = -standing_height            （贴地）

        Args:
            phase_norm: 当前阶段内的归一化进度 0~1
            is_swing: 是否在摆动相
            step_len: 步长 (m)，正=向前
            lateral_step: 侧移步长 (m)，正=向左

        Returns:
            [x, y, z] 足端位置（相对于髋关节）
        """
        step_len = np.clip(step_len, -self.step_length_max, self.step_length_max)
        lateral_step = np.clip(lateral_step, -self.step_length_max, self.step_length_max)

        if is_swing:
            s = _quintic(phase_norm)
            x = -step_len / 2 + step_len * s
            y = -lateral_step / 2 + lateral_step * s
            z = -self.standing_height + self.step_height * _asym_z(phase_norm, self.z_peak_ratio)
        else:
            x = step_len / 2 - step_len * phase_norm
            y = lateral_step / 2 - lateral_step * phase_norm
            z = -self.standing_height

        return np.array([x, y, z])
