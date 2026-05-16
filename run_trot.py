"""Go2 Trot 步态仿真 — 键盘遥操 + MuJoCo 可视化。

本文件是整个传统控制系统的入口脚本，负责：
1. 初始化 MuJoCo 仿真环境和 Go2 模型
2. 接收键盘输入，生成速度命令
3. 速度渐变（指数平滑），防止突然加减速
4. 调用 TrotController 计算目标关节角
5. PD 力矩控制跟踪目标关节角
6. MuJoCo 物理仿真步进
7. 相机跟随 + 状态显示

控制方式：
  UP       前进（按下锁定，速度渐变到最大）
  DOWN     后退
  LEFT     左转（锁定）
  RIGHT    右转（锁定）
  Q        左侧移（锁定）
  E        右侧移（锁定）
  SPACE    急停（清除所有方向锁定，速度渐变到零）
  R        重置仿真
  ESC      退出
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
import threading

from traditional_control.trot_controller import TrotController

# ============================================================
# GLFW 键盘按键编码
# ============================================================
GLFW_KEY_SPACE = 32
GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT = 263
GLFW_KEY_DOWN = 264
GLFW_KEY_UP = 265
GLFW_KEY_A = 65
GLFW_KEY_D = 68
GLFW_KEY_E = 69
GLFW_KEY_Q = 81
GLFW_KEY_R = 82
GLFW_KEY_S = 83
GLFW_KEY_W = 87
GLFW_PRESS = 1      # 按下
GLFW_RELEASE = 0    # 松开

# ============================================================
# 速度限制
# ============================================================
VX_MAX = 3.5        # 最大前进速度 (m/s)
VX_MIN = -2.5       # 最大后退速度 (m/s)
YAW_MAX = 3.5       # 最大转向角速度 (rad/s)
VY_MAX = 3.5        # 最大侧移速度 (m/s)
TAU = 0.05           # 速度渐变时间常数 (s)，越大加速越慢


class KeyState:
    """键盘按键状态追踪器。

    MuJoCo 的 launch_passive 的 key_callback 只传 (key) 一个参数，
    不支持检测松开事件。因此采用"锁定"方案：
    - 按下某个方向键后，该键保持"激活"状态
    - 按 SPACE 急停清除所有方向键
    - 按 R 重置仿真

    操作方式：按 UP 开始前进，按 SPACE 停止。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pressed = set()   # 当前激活的方向键
        self.reset = False

    # 相反方向键映射
    _OPPOSITES = {
        GLFW_KEY_UP: GLFW_KEY_DOWN, GLFW_KEY_DOWN: GLFW_KEY_UP,
        GLFW_KEY_W: GLFW_KEY_S, GLFW_KEY_S: GLFW_KEY_W,
        GLFW_KEY_LEFT: GLFW_KEY_RIGHT, GLFW_KEY_RIGHT: GLFW_KEY_LEFT,
        GLFW_KEY_A: GLFW_KEY_D, GLFW_KEY_D: GLFW_KEY_A,
        GLFW_KEY_Q: GLFW_KEY_E, GLFW_KEY_E: GLFW_KEY_Q,
    }

    def on_key(self, key: int):
        """GLFW 键盘回调函数。传给 launch_passive(key_callback=...)。"""
        with self._lock:
            if key == GLFW_KEY_R:
                self.reset = True
                self._pressed.clear()   # 重置时清除所有方向键
            elif key == GLFW_KEY_SPACE:
                self._pressed.clear()   # 急停：清除所有方向键
            else:
                # 清除相反方向键
                if key in self._OPPOSITES:
                    self._pressed.discard(self._OPPOSITES[key])
                self._pressed.add(key)  # 锁定该键

    def is_pressed(self, key: int) -> bool:
        """查询某个方向键是否处于激活状态。"""
        with self._lock:
            return key in self._pressed

    def snapshot(self) -> bool:
        """读取并清除重置标志。返回 True 表示需要重置仿真。"""
        with self._lock:
            r = self.reset
            self.reset = False
            return r


def get_body_state(data):
    """从 MuJoCo 数据中提取机体状态。

    Returns:
        body_z: 机体高度 (m)
        pitch: 俯仰角 (rad, 正=抬头)
        roll: 横滚角 (rad, 正=右倾)
        ang_vel_x: 横滚角速度 (rad/s)
        ang_vel_y: 俯仰角速度 (rad/s)
    """
    body_z = data.qpos[2]
    quat = data.qpos[3:7]  # wxyz 四元数
    w, x, y, z = quat
    # 四元数转欧拉角
    pitch = np.arcsin(np.clip(2 * (w * y - x * z), -1, 1))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    ang_vel_x = data.qvel[3]
    ang_vel_y = data.qvel[4]
    return body_z, pitch, roll, ang_vel_x, ang_vel_y


def ramp(current, target, alpha):
    """指数平滑：从 current 向 target 移动一步。

    alpha 越大响应越快，alpha=1 立即到达。
    """
    return current + (target - current) * alpha


def main():
    # --- 初始化 MuJoCo ---
    model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene.xml")
    data = mujoco.MjData(model)

    # --- 初始化控制器 ---
    controller = TrotController(freq=6.0, target_height=0.27)
    keys = KeyState()

    # PD 控制增益
    kp = 120.0   # 比例增益（刚度）
    kd = 2     # 微分增益（阻尼）

    # 当前速度（经过渐变平滑）
    vx = 0.0
    vy = 0.0
    yaw = 0.0

    print("Go2 Trot 步态 — 笛卡尔空间 IK 传统控制")
    print("操作: UP=前进  DOWN=后退  LEFT=左转  RIGHT=右转  Q=左移  E=右移  SPACE=急停  R=重置")

    with mujoco.viewer.launch_passive(model, data, key_callback=keys.on_key) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            # --- 重置检测 ---
            if keys.snapshot():
                mujoco.mj_resetDataKeyframe(model, data, 0)
                mujoco.mj_forward(model, data)
                vx = vy = yaw = 0.0
                continue

            # --- 读取按键，计算目标速度 ---
            vx_target = 0.0
            yaw_target = 0.0
            vy_target = 0.0

            if keys.is_pressed(GLFW_KEY_UP) or keys.is_pressed(GLFW_KEY_W):
                vx_target = VX_MAX
            if keys.is_pressed(GLFW_KEY_DOWN) or keys.is_pressed(GLFW_KEY_S):
                vx_target = VX_MIN
            if keys.is_pressed(GLFW_KEY_LEFT) or keys.is_pressed(GLFW_KEY_A):
                yaw_target = YAW_MAX
            if keys.is_pressed(GLFW_KEY_RIGHT) or keys.is_pressed(GLFW_KEY_D):
                yaw_target = -YAW_MAX
            if keys.is_pressed(GLFW_KEY_Q):
                vy_target = VY_MAX
            if keys.is_pressed(GLFW_KEY_E):
                vy_target = -VY_MAX

            # --- 速度渐变（指数平滑） ---
            # alpha = 1 - e^(-dt/tau)，每帧向目标速度靠近一步
            alpha = 1.0 - np.exp(-dt / TAU)
            vx = ramp(vx, vx_target, alpha)
            vy = ramp(vy, vy_target, alpha)
            yaw = ramp(yaw, yaw_target, alpha)

            # 目标为零时，速度接近零则直接归零，避免微小漂移
            if vx_target == 0.0 and abs(vx) < 0.01:
                vx = 0.0
            if vy_target == 0.0 and abs(vy) < 0.01:
                vy = 0.0
            if yaw_target == 0.0 and abs(yaw) < 0.01:
                yaw = 0.0

            # 任意方向有速度时才算行走
            walking = abs(vx) > 0.01 or abs(vy) > 0.01 or abs(yaw) > 0.01

            # --- 提取机体状态 估计器---
            body_z, pitch, roll, ang_vel_x, ang_vel_y = get_body_state(data)

            # --- 控制器计算目标关节角 ---
            # 控制器的 x 方向与世界坐标系相反，取反
            ref = controller.compute(
                t=data.time,
                vx=-vx, vy=vy, yaw_rate=yaw,
                body_z=body_z,
                pitch=pitch, roll=roll,
                ang_vel_x=ang_vel_x, ang_vel_y=ang_vel_y,
                walking=walking,
                dt=dt,
            )

            # --- PD 力矩控制 ---
            # 力矩 = kp × (目标角度 - 当前角度) - kd × 当前角速度
            n_joints = 12
            qpos = data.qpos[7:7 + n_joints]      # 关节角度（跳过前7个浮动基座）
            qvel = data.qvel[6:6 + n_joints]      # 关节角速度（跳过前6个浮动基座）
            data.ctrl[:n_joints] = kp * (ref - qpos) - kd * qvel

            # --- 物理仿真步进 ---
            mujoco.mj_step(model, data)

            # --- 相机跟随（平滑插值） ---
            alpha_cam = 0.1
            viewer.cam.lookat[:] = (1 - alpha_cam) * viewer.cam.lookat[:] + alpha_cam * data.qpos[:3]
            viewer.sync()

            # --- 状态显示 ---
            status = "[恢复中]" if controller.is_recovering else ""
            print(f"\r{status} vx={vx:+.2f} vy={vy:+.2f} yaw={yaw:+.2f}  "
                  f"pitch={np.degrees(pitch):+.1f}° roll={np.degrees(roll):+.1f}° z={body_z:.3f}m", end="")

            # --- 实时同步（保持仿真速度与现实时间一致） ---
            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
