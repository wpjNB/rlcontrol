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


# ============================================================
# 足端轨迹记录 & 可视化
# ============================================================
_TRAJECTORY_LEN = 10  # 每条腿保留的历史点数
_TRAJECTORY_STRIDE = 5  # 每隔几帧记录一次
_foot_trails = {name: [] for name in ['FL', 'FR', 'RL', 'RR']}
_frame_counter = 0


def _get_foot_world_pos(data, leg_name):
    """获取足端在世界坐标系中的位置（用 calf body 近似）。"""
    body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, f'{leg_name}_calf')
    if body_id >= 0:
        return data.xpos[body_id].copy()
    return None


def draw_visuals(viewer, data, vx, vy, show_axes=True):
    """在 MuJoCo viewer 中绘制自定义可视化。

    绘制内容：
    1. 四条腿的足端轨迹（彩色小球）
    2. 机体速度箭头（从质心出发）
    3. 关节坐标轴（红=X 绿=Y 蓝=Z）
    """
    global _frame_counter
    _frame_counter += 1
    scn = viewer.user_scn
    ngeom = 0
    eye3 = np.eye(3, dtype=np.float64).flatten()
    max_geom = len(scn.geoms)

    # --- 1. 足端轨迹（每隔几帧记录一次） ---
    trail_colors = {
        'FL': np.array([1, 0, 0, 0.8], dtype=np.float32),
        'FR': np.array([0, 1, 0, 0.8], dtype=np.float32),
        'RL': np.array([0, 0, 1, 0.8], dtype=np.float32),
        'RR': np.array([1, 1, 0, 0.8], dtype=np.float32),
    }
    record = (_frame_counter % _TRAJECTORY_STRIDE == 0)
    for leg_name in ['FL', 'FR', 'RL', 'RR']:
        if record:
            foot_pos = _get_foot_world_pos(data, leg_name)
            if foot_pos is not None:
                _foot_trails[leg_name].append(foot_pos.copy())
                if len(_foot_trails[leg_name]) > _TRAJECTORY_LEN:
                    _foot_trails[leg_name].pop(0)

        trail = _foot_trails[leg_name]
        color = trail_colors[leg_name]
        n_pts = len(trail)
        for i, pos in enumerate(trail):
            if ngeom >= max_geom:
                break
            g = scn.geoms[ngeom]
            alpha = color[3] * ((i + 1) / max(n_pts, 1))
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.005, 0, 0], dtype=np.float64),
                pos.astype(np.float64), eye3,
                np.array([color[0], color[1], color[2], alpha], dtype=np.float32),
            )
            ngeom += 1

    # --- 2. 速度箭头 ---
    body_pos = data.qpos[:3].copy()
    speed = np.sqrt(vx**2 + vy**2)
    if speed > 0.01 and ngeom < max_geom:
        arrow_end = body_pos.copy()
        arrow_end[0] += vx * 0.3
        arrow_end[1] += vy * 0.3
        direction = arrow_end - body_pos
        d_norm = direction / (np.linalg.norm(direction) + 1e-8)
        up = np.array([0, 0, 1], dtype=np.float64)
        if abs(np.dot(d_norm, up)) > 0.99:
            up = np.array([0, 1, 0], dtype=np.float64)
        right = np.cross(d_norm, up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, d_norm)
        rot = np.column_stack([d_norm, right, up]).flatten()

        g = scn.geoms[ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([0.008, speed * 0.15, 0], dtype=np.float64),
            ((body_pos + arrow_end) / 2).astype(np.float64), rot,
            np.array([1, 0.5, 0, 1], dtype=np.float32),
        )
        ngeom += 1

    # --- 3. 关节坐标轴（红=X 绿=Y 蓝=Z） ---
    if show_axes:
        axis_len = 0.04  # 轴长度 4cm
        axis_colors = [
            np.array([1, 0, 0, 1], dtype=np.float32),  # X 红
            np.array([0, 1, 0, 1], dtype=np.float32),  # Y 绿
            np.array([0, 0, 1, 1], dtype=np.float32),  # Z 蓝
        ]
        for leg in ['FL', 'FR', 'RL', 'RR']:
            for part in ['hip', 'thigh', 'calf']:
                body_id = mujoco.mj_name2id(
                    data.model, mujoco.mjtObj.mjOBJ_BODY, f'{leg}_{part}')
                if body_id < 0:
                    continue
                pos = data.xpos[body_id].astype(np.float64)
                mat = data.xmat[body_id].reshape(3, 3)  # 旋转矩阵
                for axis_idx in range(3):  # X, Y, Z
                    if ngeom >= max_geom:
                        break
                    axis_dir = mat[:, axis_idx]  # 世界坐标系下的轴方向
                    g = scn.geoms[ngeom]
                    # 用 capsule 表示轴：半径 1mm, 半长 = axis_len/2
                    # capsule 局部轴是 x 轴，需要旋转对齐
                    end = pos + axis_dir * axis_len
                    center = (pos + end) / 2
                    # 构造旋转矩阵让局部 x 轴对齐 axis_dir
                    up = np.array([0, 0, 1], dtype=np.float64)
                    if abs(np.dot(axis_dir, up)) > 0.99:
                        up = np.array([0, 1, 0], dtype=np.float64)
                    right = np.cross(axis_dir, up)
                    right /= np.linalg.norm(right) + 1e-8
                    up = np.cross(right, axis_dir)
                    rot_mat = np.column_stack([axis_dir, right, up]).flatten()
                    mujoco.mjv_initGeom(
                        g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.array([0.001, axis_len / 2, 0], dtype=np.float64),
                        center, rot_mat,
                        axis_colors[axis_idx],
                    )
                    ngeom += 1

    scn.ngeom = ngeom


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

        # 可视化：接触力箭头


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

            # --- 绘制可视化（轨迹 + 箭头 + 坐标轴） ---
            draw_visuals(viewer, data, vx, vy, show_axes=True)
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
