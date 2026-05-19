"""轨迹可视化 — 绘制参考轨迹、跟踪路径和速度矢量。

使用 matplotlib 生成三张子图：
1. XY 平面轨迹 + 速度矢量
2. 航向角 yaw 随时间变化
3. 速度分量 (vx, vy, v_yaw) 随时间变化
"""

import numpy as np
import matplotlib.pyplot as plt
from trajectory import TrajPoint


def plot_trajectory(
    ref_traj: list[TrajPoint],
    track_path: list[tuple[float, float]] | None = None,
    title: str = "Trajectory",
    save_path: str | None = None,
):
    """绘制参考轨迹和可选的跟踪路径。

    Args:
        ref_traj: 参考轨迹点列表
        track_path: 实际跟踪路径 [(x, y), ...]，可选
        title: 图标题
        save_path: 保存图片路径，None 则直接显示
    """
    xs = [p.x for p in ref_traj]
    ys = [p.y for p in ref_traj]
    ts = [p.t for p in ref_traj]
    yaws = [p.yaw for p in ref_traj]
    vxs = [p.vx for p in ref_traj]
    vys = [p.vy for p in ref_traj]
    vyaws = [p.v_yaw for p in ref_traj]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14)

    # --- 子图 1: XY 轨迹 ---
    ax = axes[0]
    ax.plot(xs, ys, 'b-', linewidth=1.5, label='Reference')
    ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
    ax.plot(xs[-1], ys[-1], 'rs', markersize=10, label='End')

    # 速度矢量（每隔 N 个点画一个箭头）
    step = max(1, len(ref_traj) // 30)
    for i in range(0, len(ref_traj), step):
        p = ref_traj[i]
        ax.annotate('', xy=(p.x + p.vx * 0.3, p.y + p.vy * 0.3), xytext=(p.x, p.y),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # 航向角指示（每隔 M 个点画一个箭头）
    step_yaw = max(1, len(ref_traj) // 15)
    arrow_len = 0.4
    for i in range(0, len(ref_traj), step_yaw):
        p = ref_traj[i]
        dx = arrow_len * np.cos(p.yaw)
        dy = arrow_len * np.sin(p.yaw)
        ax.annotate('', xy=(p.x + dx, p.y + dy), xytext=(p.x, p.y),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.0))

    if track_path:
        tx = [p[0] for p in track_path]
        ty = [p[1] for p in track_path]
        ax.plot(tx, ty, 'r--', linewidth=1.0, alpha=0.7, label='Tracked')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- 子图 2: 航向角 ---
    ax = axes[1]
    ax.plot(ts, np.degrees(yaws), 'r-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw (deg)')
    ax.grid(True, alpha=0.3)

    # --- 子图 3: 速度 ---
    ax = axes[2]
    ax.plot(ts, vxs, 'b-', linewidth=1.2, label='vx')
    ax.plot(ts, vys, 'g-', linewidth=1.2, label='vy')
    ax.plot(ts, vyaws, 'r-', linewidth=1.2, label='v_yaw')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def simulate_tracking(
    ref_traj: list[TrajPoint],
    dt: float = 0.06,
    vx_gain: float = 1.0,
    vy_gain: float = 1.0,
    noise_std: float = 0.0,
) -> list[tuple[float, float]]:
    """简单模拟轨迹跟踪，用于可视化验证。

    用一阶积分模型模拟机器人跟踪参考轨迹，可加噪声测试鲁棒性。

    Args:
        ref_traj: 参考轨迹
        dt: 时间步长 (s)
        vx_gain, vy_gain: 速度跟踪增益（模拟跟踪延迟）
        noise_std: 位置噪声标准差 (m)，0 表示无噪声

    Returns:
        实际路径 [(x, y), ...]
    """
    from follower import TrajectoryFollower

    follower = TrajectoryFollower(lookahead_dist=0.3)
    x, y = ref_traj[0].x, ref_traj[0].y
    path = [(x, y)]

    for _ in range(len(ref_traj)):
        cmd, done = follower.track(ref_traj, x, y)
        x += cmd['vx'] * vx_gain * dt 
        y += cmd['vy'] * vy_gain * dt 
        path.append((x, y))
        if done:
            break

    return path


if __name__ == '__main__':
    from trajectory import gen_circle_traj, gen_sin_traj

    # 圆形轨迹
    circle = gen_circle_traj(point_num=200, dt=0.06, radius=3.0, omega=0.25)
    circle_path = simulate_tracking(circle, dt=0.06, noise_std=0.02)
    plot_trajectory(circle, track_path=circle_path, title="Circle Trajectory (r=3m, w=0.25 rad/s)")

    # 正弦轨迹
    sin_traj = gen_sin_traj(point_num=200, dt=0.06, amp=1.0, vx=0.5)
    sin_path = simulate_tracking(sin_traj, dt=0.06, noise_std=0.02)
    plot_trajectory(sin_traj, track_path=sin_path, title="Sinusoidal Trajectory (amp=1m, vx=0.5 m/s)")
