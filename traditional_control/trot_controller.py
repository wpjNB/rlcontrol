"""Trot 步态主控制器 — 组合相位调度、足端轨迹、逆运动学、平衡补偿和跌倒恢复。

本模块是传统控制的核心，每帧调用 compute() 完成：
1. 跌倒检测：倾斜角过大时进入恢复模式
2. 相位调度：计算每条腿当前处于摆动/支撑相
3. 足端轨迹：在笛卡尔空间规划每条腿的足端目标位置
4. 逆运动学：将足端位置转为关节角度
5. 姿态补偿：根据 pitch/roll/height 误差修正关节角
6. 输出 12 维目标关节角

输入：速度命令 (vx, vy, yaw_rate) + 机体状态 (body_z, pitch, roll, ang_vel)
输出：12 维目标关节角 [髋,大腿,小腿] × [FL,FR,RL,RR]
"""

import numpy as np
from traditional_control.kinematics import Go2LegKinematics, HOME_QPOS, LEGS
from traditional_control.gait_phase import GaitPhaseScheduler
from traditional_control.foot_trajectory import FootTrajectory

# 每条腿在 12 维关节角数组中的起始索引
LEG_INDICES = {'FL': 0, 'FR': 3, 'RL': 6, 'RR': 9}

# 腿分组（用于姿态补偿）
FRONT_LEGS = ['FL', 'FR']   # 前腿
REAR_LEGS = ['RL', 'RR']    # 后腿
LEFT_LEGS = ['FL', 'RL']    # 左侧腿
RIGHT_LEGS = ['FR', 'RR']   # 右侧腿


class TrotController:
    """Trot 步态主控制器。

    Args:
        freq: 步频 Hz（默认 2.0）
        duty_cycle: 支撑相比例（默认 0.6）
        step_height: 抬腿高度 m（默认 0.06）
        target_height: 目标机体高度 m（默认 0.27）
        tilt_limit: 触发跌倒恢复的最大倾斜角 rad（默认 0.35 ≈ 20°）
    """

    def __init__(
        self,
        freq: float = 2.0,
        duty_cycle: float = 0.6,
        step_height: float = 0.10,
        target_height: float = 0.27,
        tilt_limit: float = 0.35,
    ):
        self.kin = Go2LegKinematics()                              # 运动学求解器
        self.phase = GaitPhaseScheduler(freq=freq, duty_cycle=duty_cycle)  # 相位调度
        self.trajectory = FootTrajectory(step_height=step_height)  # 足端轨迹
        self.target_height = target_height
        self.tilt_limit = tilt_limit

        # 姿态补偿 PD 增益
        self.kp_pitch = 6    # 俯仰补偿增益（临时调大验证极性）
        self.kp_roll = 0.5     # 横滚补偿增益
        self.kp_height = 5.0   # 高度补偿增益

        # 跌倒恢复状态
        self._recovering = False
        self._recover_timer = 0.0
        self._recover_duration = 0.5  # 恢复姿态持续时间（秒）

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
        """主控制函数：速度命令 + 机体状态 → 12 维目标关节角。

        处理流程：
        1. 跌倒检测 → 倾斜过大则进入恢复模式
        2. 恢复模式 → 返回蹲伏姿态
        3. 站立模式 → 返回站立关节角
        4. 步态模式 → 相位调度 + 足端轨迹 + IK + 平衡补偿

        Args:
            t: 当前时间 (s)
            vx, vy, yaw_rate: 速度命令
            body_z: 机体高度 (m)
            pitch, roll: 机体俯仰/横滚角 (rad)
            ang_vel_x, ang_vel_y: 角速度 (rad/s)
            walking: 是否在行走
            dt: 时间步长 (s)

        Returns:
            12d 目标关节角
        """
        # --- 跌倒检测 ---
        # 倾斜角 = sqrt(pitch² + roll²)，超过阈值则触发恢复
        tilt = np.sqrt(pitch**2 + roll**2)
        ang_vel_mag = np.sqrt(ang_vel_x**2 + ang_vel_y**2)

        if not self._recovering and (
            tilt > self.tilt_limit or (tilt > 0.2 and ang_vel_mag > 2.0)
        ):
            self._recovering = True
            self._recover_timer = 0.0

        # --- 恢复模式：蹲伏 + 反向补偿 ---
 

        # --- 站立模式：站立关节角 ---
        if not walking:
            ref = HOME_QPOS.copy()
        else:
            # --- 步态模式：相位调度 + 足端轨迹 + IK ---
            phases = self.phase.step(t, dt)
            lateral_step = self.phase.get_lateral_offset(vy)
            ref = np.zeros(12)

            for leg in LEGS:
                idx = LEG_INDICES[leg]
                p = phases[leg]
                step_len = self.phase.get_step_length(vx, yaw_rate, leg)
                foot_pos = self.trajectory.compute(
                    phase_norm=p['phase_norm'],
                    is_swing=p['is_swing'],
                    step_len=step_len,
                    lateral_step=lateral_step,
                )
                joints = self.kin.solve(leg, foot_pos)
                joints = self.kin.clamp_joints(leg, joints)
                ref[idx:idx + 3] = joints

        # --- 姿态平衡补偿（站立和步态都生效） ---
        ref = self._apply_balance(ref, pitch, roll, body_z, ang_vel_x, ang_vel_y)
        return ref

    def _apply_balance(self, ref, pitch, roll, body_z, ang_vel_x, ang_vel_y):
        """PD 姿态平衡补偿。

        根据机体的俯仰、横滚和高度误差，微调关节角：
        - 俯仰补偿：前倾→前腿伸展后腿收缩，后倾→反之
        - 横滚补偿：右倾→右腿伸展左腿收缩，左倾→反之
        - 高度补偿：太低→所有腿伸展，太高→所有腿收缩
        """
        # 俯仰补偿（比例 + 微分）
        # pitch_corr = self.kp_pitch * pitch
        # for leg in FRONT_LEGS:
        #     idx = LEG_INDICES[leg]
        #     ref[idx + 1] -= pitch_corr * 0.3   # 前腿大腿
        #     ref[idx + 2] += pitch_corr * 0.4   # 前腿小腿

        # for leg in REAR_LEGS:
        #     idx = LEG_INDICES[leg]
        #     ref[idx + 1] += pitch_corr * 0.3   # 后腿大腿
        #     ref[idx + 2] -= pitch_corr * 0.4

        # 横滚补偿
        roll_corr = self.kp_roll * roll
        for leg in RIGHT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= roll_corr * 0.2    # 右腿大腿
            ref[idx + 0] -= roll_corr * 0.1    # 右腿髋
        for leg in LEFT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += roll_corr * 0.2    # 左腿大腿
            ref[idx + 0] += roll_corr * 0.1

        # 高度补偿
        height_err = self.target_height - body_z
        height_corr = self.kp_height * height_err
        for leg in LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= height_corr * 0.2  # 大腿伸展/收缩
            ref[idx + 2] += height_corr * 0.15 # 小腿补偿

        return ref

    def _recovery_pose(self, pitch, roll):
        """跌倒恢复姿态：蹲伏 + 反向倾斜补偿。

        策略：降低重心（弯曲小腿）+ 反向补偿倾斜角，帮助机器人恢复平衡。
        """
        ref = HOME_QPOS.copy()

        # 蹲伏：所有小腿额外弯曲 0.3rad
        for leg in LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 2] -= 0.3

        # 俯仰反向补偿
        pitch_corr = 0.8 * pitch
        for leg in FRONT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= pitch_corr * 0.4
        for leg in REAR_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += pitch_corr * 0.4

        # 横滚反向补偿
        roll_corr = 0.6 * roll
        for leg in RIGHT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= roll_corr * 0.3
        for leg in LEFT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += roll_corr * 0.3

        return ref

    @property
    def is_recovering(self) -> bool:
        """是否正在跌倒恢复中。"""
        return self._recovering
