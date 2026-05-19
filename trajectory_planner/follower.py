"""轨迹跟踪器 — 基于最近点的纯追踪（Pure Pursuit）跟踪策略。

从参考轨迹中找到离当前位置最近的前向点，计算速度和航向跟踪误差。
"""

import numpy as np
from trajectory import TrajPoint


class TrajectoryFollower:
    """轨迹跟踪控制器。

    策略：
    1. 在参考轨迹中找到当前位置前方最近的前瞻点（lookahead）
    2. 输出该点的速度命令 (vx, vy, v_yaw)
    3. 判断是否到达轨迹终点

    Args:
        lookahead_dist: 前瞻距离 (m)，越大越平滑但跟踪延迟越大
        reach_thresh: 到达终点判定距离 (m)
    """

    def __init__(self, lookahead_dist: float = 0.5, reach_thresh: float = 0.3):
        self.lookahead_dist = lookahead_dist
        self.reach_thresh = reach_thresh
        self._idx = 0  # 当前最近点索引

    def reset(self):
        """重置跟踪器状态。"""
        self._idx = 0

    def track(
        self,
        ref_traj: list[TrajPoint],
        cur_x: float,
        cur_y: float,
    ) -> tuple[dict, bool]:
        """执行一步轨迹跟踪。

        Args:
            ref_traj: 参考轨迹
            cur_x, cur_y: 当前机器人位置

        Returns:
            (cmd, done):
                cmd: {'vx', 'vy', 'v_yaw', 'target_x', 'target_y', 'target_yaw'}
                done: 是否到达终点
        """
        if not ref_traj:
            return {'vx': 0, 'vy': 0, 'v_yaw': 0, 'target_x': 0, 'target_y': 0, 'target_yaw': 0}, True

        # 从当前索引向前搜索前瞻点
        min_dist = float('inf')
        best_idx = self._idx

        for i in range(self._idx, len(ref_traj)):
            p = ref_traj[i]
            dist = np.hypot(p.x - cur_x, p.y - cur_y)
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        # 向前推进到满足前瞻距离的点
        target_idx = best_idx
        for i in range(best_idx, len(ref_traj)):
            p = ref_traj[i]
            dist = np.hypot(p.x - cur_x, p.y - cur_y)
            if dist >= self.lookahead_dist:
                target_idx = i
                break
        else:
            target_idx = len(ref_traj) - 1

        self._idx = best_idx
        target = ref_traj[target_idx]

        # 检查是否到达终点
        last = ref_traj[-1]
        done = np.hypot(last.x - cur_x, last.y - cur_y) < self.reach_thresh

        cmd = {
            'vx': target.vx,
            'vy': target.vy,
            'v_yaw': target.v_yaw,
            'target_x': target.x,
            'target_y': target.y,
            'target_yaw': target.yaw,
        }
        return cmd, done
