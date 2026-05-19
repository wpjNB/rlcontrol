"""轨迹生成器 — 圆形和正弦参考轨迹。

提供 TrajPoint 数据结构和两种轨迹生成函数，输出带时间戳的参考轨迹点序列。

已修正原 C++ 版本的 bug：
- 圆形轨迹 y 分量原误用 cos，现改为 sin
- 正弦轨迹 vy 和 v_yaw 导数计算已修正
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class TrajPoint:
    """轨迹点：位置、速度、航向角、时间戳。"""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    v_yaw: float = 0.0
    t: float = 0.0


def gen_circle_traj(
    point_num: int,
    dt: float = 0.06,
    radius: float = 3.0,
    omega: float = 0.25,
) -> list[TrajPoint]:
    """生成圆形参考轨迹。

    机器人以角速度 omega 绕半径 radius 的圆周运动。

    x = r * cos(omega * t)
    y = r * sin(omega * t)
    yaw = omega * t

    Args:
        point_num: 轨迹点数量
        dt: 时间间隔 (s)
        radius: 圆半径 (m)
        omega: 航向角速度 (rad/s)

    Returns:
        TrajPoint 列表
    """
    traj: list[TrajPoint] = []
    for i in range(point_num):
        t = i * dt
        theta = omega * t
        p = TrajPoint(
            x=radius * np.cos(theta),
            y=radius * np.sin(theta),       # 修正：原 C++ 误用 cos
            yaw=theta,
            vx=-radius * omega * np.sin(theta),
            vy=radius * omega * np.cos(theta),
            v_yaw=omega,
            t=t,
        )
        traj.append(p)
    return traj


def gen_sin_traj(
    point_num: int,
    dt: float = 0.06,
    amp: float = 1.0,
    vx: float = 0.5,
) -> list[TrajPoint]:
    """生成正弦参考轨迹。

    机器人以恒定 vx 前进，横向位移 y = amp * sin(x)。

    x  = vx * t
    y  = amp * sin(vx * t)
    yaw = arctan(dy/dx) = arctan(amp * cos(vx * t))

    Args:
        point_num: 轨迹点数量
        dt: 时间间隔 (s)
        amp: 正弦振幅 (m)
        vx: 前进速度 (m/s)

    Returns:
        TrajPoint 列表
    """
    traj: list[TrajPoint] = []
    for i in range(point_num):
        t = i * dt
        x = vx * t
        cos_vxt = np.cos(vx * t)
        sin_vxt = np.sin(vx * t)

        dydx = amp * cos_vxt                    # dy/dx = amp * cos(vx*t)
        d2ydx2 = -amp * sin_vxt                 # d²y/dx² = -amp * sin(vx*t)

        p = TrajPoint(
            x=x,
            y=amp * sin_vxt,
            vx=vx,
            vy=amp * vx * cos_vxt,              # 修正：原 C++ 误用 sin
            yaw=np.arctan(dydx),
            v_yaw=d2ydx2 / (1 + dydx**2) * vx,  # 修正：d/dt(arctan(dy/dx)) * dx/dt
            t=t,
        )
        traj.append(p)
    return traj
