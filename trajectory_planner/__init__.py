"""轨迹规划器 — 生成参考轨迹并跟踪执行。

模块结构：
- trajectory.py   : TrajPoint 数据结构 + 圆形/正弦轨迹生成
- follower.py     : 最近点轨迹跟踪器
- visualize.py    : matplotlib 可视化
"""

from trajectory_planner.trajectory import TrajPoint, gen_circle_traj, gen_sin_traj
from trajectory_planner.follower import TrajectoryFollower

__all__ = ['TrajPoint', 'gen_circle_traj', 'gen_sin_traj', 'TrajectoryFollower']
