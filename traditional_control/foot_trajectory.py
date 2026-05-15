"""笛卡尔空间足端轨迹生成器 — 规划每条腿在摆动/支撑相的足端运动轨迹。

本模块将步态相位转化为具体的足端位置目标：
- 摆动相（脚离地）：正弦曲线抬腿 + 线性前摆
- 支撑相（脚踩地）：足端固定不动，身体从脚上方移过

输出的足端位置 [x, y, z] 是相对于髋关节的，后续由 IK 模块转为关节角。
"""

import numpy as np


class FootTrajectory:
    """笛卡尔空间足端轨迹生成器。

    Args:
        step_height: 最大抬腿高度 (m)，默认 0.06
        step_length_max: 最大步长 (m)，用于裁剪，默认 0.2
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
        """计算足端目标位置（相对于髋关节）。

        摆动相轨迹：
          x = -step_len/2 → +step_len/2  （从前到后线性前摆）
          z = step_height * sin(π * t)    （正弦抬腿，中间最高）

        支撑相轨迹：
          x = +step_len/2 → -step_len/2  （足端固定，身体前移）
          z = 0                           （贴地）

        Args:
            phase_norm: 当前阶段内的归一化进度 0~1
            is_swing: 是否在摆动相
            step_len: 步长 (m)，正=向前
            vy_offset: 横向偏移 (m)

        Returns:
            [x, y, z] 足端位置（相对于髋关节）
        """
        # 裁剪步长到最大值，防止步子过大
        step_len = np.clip(step_len, -self.step_length_max, self.step_length_max)

        if is_swing:
            # 摆动相：线性前摆 + 正弦抬腿
            x = -step_len / 2 + step_len * phase_norm
            z = self.step_height * np.sin(np.pi * phase_norm)
        else:
            # 支撑相：足端固定，身体前移
            x = step_len / 2 - step_len * phase_norm
            z = 0.0

        y = vy_offset

        return np.array([x, y, z])
