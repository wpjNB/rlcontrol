"""五次多项式轨迹规划示例。

约束：起点 A 和终点 B 的位置、速度、加速度共 6 个条件，
恰好由 5 次多项式的 6 个系数唯一确定。

    p(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵

    边界条件：
        p(0)   = p0      p(T)   = p1
        p'(0)  = v0      p'(T)  = v1
        p''(0) = a0      p''(T) = a1
"""

import numpy as np
import matplotlib.pyplot as plt


def quintic_coeff(p0, v0, acc0, p1, v1, acc1, T):
    """计算五次多项式系数 [a0, a1, a2, a3, a4, a5]。

    通过求解边界条件构成的 6×6 线性方程组得到。
    """
    # 构造矩阵 M @ coeffs = b
    # p(t)  = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
    # p'(t) = a1 + 2*a2*t + 3*a3*t² + 4*a4*t³ + 5*a5*t⁴
    # p''(t)= 2*a2 + 6*a3*t + 12*a4*t² + 20*a5*t³
    M = np.array([
        [1, 0,    0,      0,      0,       0],       # p(0)
        [0, 1,    0,      0,      0,       0],       # p'(0)
        [0, 0,    2,      0,      0,       0],       # p''(0)
        [1, T,    T**2,   T**3,   T**4,    T**5],    # p(T)
        [0, 1,    2*T,    3*T**2, 4*T**3,  5*T**4],  # p'(T)
        [0, 0,    2,      6*T,    12*T**2, 20*T**3], # p''(T)
    ])
    b = np.array([p0, v0, acc0, p1, v1, acc1])
    return np.linalg.solve(M, b)


def quintic_eval(coeffs, t):
    """计算多项式值、一阶导、二阶导。"""
    a0, a1, a2, a3, a4, a5 = coeffs
    p   = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    dp  = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    ddp = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    return p, dp, ddp


def plan_quintic_2d(p0, v0, acc0, p1, v1, acc1, T, n=200):
    """2D 五次多项式轨迹规划。

    Args:
        p0:   起点位置 (x, y)
        v0:   起点速度 (vx, vy)
        acc0: 起点加速度 (ax, ay)
        p1:   终点位置
        v1:   终点速度
        acc1: 终点加速度
        T:    总时间
        n:    采样点数

    Returns:
        t, pos, vel, acc — 各为 (n, 2) 数组
    """
    cx = quintic_coeff(p0[0], v0[0], acc0[0], p1[0], v1[0], acc1[0], T)
    cy = quintic_coeff(p0[1], v0[1], acc0[1], p1[1], v1[0], acc1[1], T)

    t = np.linspace(0, T, n)
    px, vx, ax = quintic_eval(cx, t)
    py, vy, ay = quintic_eval(cy, t)

    pos = np.column_stack([px, py])
    vel = np.column_stack([vx, vy])
    acc = np.column_stack([ax, ay])
    return t, pos, vel, acc


def plot_quintic_demo():
    """演示：从 A 到 B 的五次多项式轨迹。"""
    # ---- 边界条件 ----
    p0   = np.array([0.0, 0.0])     # 起点 A
    v0   = np.array([0.0, 0.0])     # 起点速度
    acc0 = np.array([0.0, 0.0])     # 起点加速度

    p1   = np.array([3.0, 2.0])     # 终点 B
    v1   = np.array([0.5, 0.3])     # 终点速度（非零 = 有末速度要求）
    acc1 = np.array([0.0, 0.0])     # 终点加速度

    T = 4.0  # 总时间 (s)

    # ---- 规划 ----
    t, pos, vel, acc = plan_quintic_2d(p0, v0, acc0, p1, v1, acc1, T)

    # ---- 绘图 ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Quintic Polynomial Trajectory Planning', fontsize=14, fontweight='bold')

    # 1) XY 轨迹 + 速度矢量
    ax = axes[0, 0]
    ax.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=2, label='trajectory')
    ax.plot(*p0, 'go', markersize=12, zorder=5, label='A (start)')
    ax.plot(*p1, 'rs', markersize=12, zorder=5, label='B (end)')
    # 起终点速度箭头
    ax.annotate('', xy=p0 + v0 * 0.8, xytext=p0,
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=p1 + v1 * 0.8, xytext=p1,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    # 轨迹上速度矢量
    step = len(t) // 20
    for i in range(0, len(t), step):
        ax.annotate('', xy=pos[i] + vel[i] * 0.3, xytext=pos[i],
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title('XY Trajectory')

    # 2) X, Y 位置随时间
    ax = axes[0, 1]
    ax.plot(t, pos[:, 0], 'b-', linewidth=1.5, label='x(t)')
    ax.plot(t, pos[:, 1], 'r-', linewidth=1.5, label='y(t)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Position vs Time')

    # 3) 速度
    ax = axes[0, 2]
    speed = np.linalg.norm(vel, axis=1)
    ax.plot(t, vel[:, 0], 'b-', linewidth=1.2, label='vx')
    ax.plot(t, vel[:, 1], 'r-', linewidth=1.2, label='vy')
    ax.plot(t, speed, 'k--', linewidth=1.2, label='|v|', alpha=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Velocity vs Time')

    # 4) 加速度
    ax = axes[1, 0]
    ax.plot(t, acc[:, 0], 'b-', linewidth=1.2, label='ax')
    ax.plot(t, acc[:, 1], 'r-', linewidth=1.2, label='ay')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Acceleration vs Time')

    # 5) Jerk（加加速度）
    ax = axes[1, 1]
    dt = t[1] - t[0]
    jerk_x = np.gradient(acc[:, 0], dt)
    jerk_y = np.gradient(acc[:, 1], dt)
    ax.plot(t, jerk_x, 'b-', linewidth=1.2, label='jx')
    ax.plot(t, jerk_y, 'r-', linewidth=1.2, label='jy')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Jerk (m/s³)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Jerk vs Time')

    # 6) 约束验证表
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['', 'X', 'Y'],
        ['p(0)', f'{pos[0,0]:.3f}', f'{pos[0,1]:.3f}'],
        ['p(T)', f'{pos[-1,0]:.3f}', f'{pos[-1,1]:.3f}'],
        ['v(0)', f'{vel[0,0]:.3f}', f'{vel[0,1]:.3f}'],
        ['v(T)', f'{vel[-1,0]:.3f}', f'{vel[-1,1]:.3f}'],
        ['a(0)', f'{acc[0,0]:.3f}', f'{acc[0,1]:.3f}'],
        ['a(T)', f'{acc[-1,0]:.3f}', f'{acc[-1,1]:.3f}'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    # 表头样式
    for j in range(3):
        table[0, j].set_facecolor('#2a2a3a')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('Boundary Constraints (Actual)', pad=20)

    plt.tight_layout()
    plt.savefig('trajectory_planner/quintic_demo.png', dpi=150, bbox_inches='tight')
    print('Saved: trajectory_planner/quintic_demo.png')
    plt.show()


if __name__ == '__main__':
    plot_quintic_demo()
