# Go1 自定义强化学习环境设计

## 概述

基于 Unitree Go1 四足机器人 MuJoCo 模型，设计模块化的 Gymnasium 自定义环境，支持三个递进任务：平衡站立、前进移动、跳跃过障碍。使用 stable-baselines3 PPO 训练。

## 架构

### 方案选择

自定义 `gymnasium.Env` 子类 + 共享基类 `Go1BaseEnv`，而非基于 Ant-v5 的 Wrapper 方案。理由：
- Ant-v5 的观测/奖励结构为 "Ant" 设计，Go1 关节结构和控制目标不同
- 自定义 Env 可精确控制每个任务的奖励函数、观测空间、终止条件
- 共享基类避免重复代码，子类只需覆盖差异部分

### 目录结构

```
go1_envs/
├── __init__.py              # 包导出
├── base.py                  # Go1BaseEnv 基类
├── go1_balance.py           # Go1BalanceEnv: 平衡站立
├── go1_walk.py              # Go1WalkEnv: 前进移动
├── go1_jump.py              # Go1JumpEnv: 跳跃过障碍
├── train.py                 # 统一训练入口
└── scenes/
    ├── flat_scene.xml       # 平坦地面（平衡/行走）
    └── obstacle_scene.xml   # 带障碍物地面（跳跃）
```

## 基类 Go1BaseEnv

### 动作空间

`Box(shape=(12,), dtype=float32)` — 12 个关节的目标角度，范围由 `go1.xml` 的 `ctrlrange` 定义：
- FR_hip, FR_thigh, FR_calf (前右腿 3 关节)
- FL_hip, FL_thigh, FL_calf (前左腿 3 关节)
- RR_hip, RR_thigh, RR_calf (后右腿 3 关节)
- RL_hip, RL_thigh, RL_calf (后左腿 3 关节)

### 观测空间

`Box(shape=(51,), dtype=float32)` — 基础观测，子类可扩展：
- `qpos[2:21]` (19维): 关节位置（去除全局 x,y 位置，保留 z 高度 + 四元数 + 12 关节角）
- `qvel[0:18]` (18维): 关节速度（6 全局速度 + 12 关节角速度）
- `body_quat` (4维): 机身四元数（冗余但方便网络学习）
- `body_vel` (6维): 机身线速度 + 角速度
- `foot_contacts` (4维): 四条腿是否触地（通过接触力判断）

### 子类钩子方法

子类必须覆盖：
- `_get_reward(obs, action) -> float`: 奖励计算
- `_is_terminated() -> bool`: 终止条件

子类可选覆盖：
- `_get_info() -> dict`: 额外信息（用于日志/调试）
- `_reset_noise() -> ndarray`: 自定义重置噪声
- `_get_obs() -> ndarray`: 扩展观测（调用 `super()._get_obs()` 后拼接额外信息）

### 通用逻辑

- **前向仿真**: `mj_step` × `frame_skip`（默认 frame_skip=25, dt=0.05s）
- **通用终止**: 机身 z < 0.15m（摔倒）
- **重置**: 从 home 关键帧 + 高斯噪声重置

## 任务 1: Go1BalanceEnv — 平衡站立

**目标**: Go1 保持站立姿态，抵抗随机扰动。

### 奖励函数

| 项 | 权重 | 说明 |
|---|---|---|
| 存活奖励 | +1.0 | 每步站立 |
| 摔倒惩罚 | -10.0 | 机身 z < 0.15m |
| 动作代价 | -0.1 | `‖action‖²`，鼓励节能 |
| 倾斜惩罚 | -0.5 | `‖body_tilt‖²`，鼓励直立 |

### 终止条件

- 机身高度 z < 0.15m
- 机身倾斜角 > 45°

### 扰动机制

每 100~200 步（随机间隔），在机身施加 10~30N 的随机方向水平力，持续 10 步。通过 `data.xfrc_applied` 在代码中实现，无需修改 XML。

### 场景

`flat_scene.xml` — 基于现有 `scene.xml`，平坦地面 + Go1。

## 任务 2: Go1WalkEnv — 前进移动

**目标**: Go1 向 x 轴正方向行走。

### 奖励函数

| 项 | 权重 | 说明 |
|---|---|---|
| 前进速度 | +1.0 | x 方向速度 |
| 存活奖励 | +1.0 | 每步存活 |
| 控制代价 | -0.05 | `‖action‖²` |
| 横向惩罚 | -0.1 | y 方向速度²，惩罚偏移 |

### 终止条件

- 摔倒（z < 0.15m）
- y 偏移 > 2m
- x > 20m（到达终点）

### 场景

`flat_scene.xml`

## 任务 3: Go1JumpEnv — 跳跃过障碍

**目标**: Go1 跳过前方障碍物。

### 奖励函数

| 项 | 权重 | 说明 |
|---|---|---|
| 跳跃高度 | +10.0 | `max(0, body_z - obstacle_top)` |
| 前进速度 | +1.0 | x 方向速度 |
| 碰撞惩罚 | -100.0 | 接触障碍物 |
| 越过奖励 | +50.0 | 成功越过障碍物（x > obstacle_x 且未碰撞） |

### 终止条件

- 摔倒
- 碰撞障碍物（接触力检测）
- x > 5m（到达终点）

### 场景

`obstacle_scene.xml` — 在 `flat_scene.xml` 基础上增加障碍物：
```xml
<body name="obstacle" pos="2 0 0.1">
  <geom type="box" size="0.05 0.5 0.1" rgba="0.8 0.2 0.2 1"/>
</body>
```

### 关键帧

增加下蹲准备姿态：
```xml
<key name="crouch" qpos="0 0 0.20 1 0 0 0 0 1.2 -2.2 0 1.2 -2.2 0 1.2 -2.2 0 1.2 -2.2"/>
```

## 训练流水线

### 统一入口 `train.py`

```bash
python -m go1_envs.train balance --timesteps 300000 --n-envs 8
python -m go1_envs.train walk --timesteps 500000 --n-envs 8
python -m go1_envs.train jump --timesteps 800000 --n-envs 8
```

### PPO 超参数

复用 `load_quadruped_model.py` 中验证过的参数：
- learning_rate=3e-4
- n_steps=2048
- batch_size=512
- n_epochs=5
- gamma=0.99
- gae_lambda=0.95
- clip_range=0.2
- ent_coef=0.001

### 训练加速

- SubprocVecEnv 多进程并行（默认 8 环境）
- VecNormalize 自动归一化观测和奖励
- EvalCallback 定期评估 + 保存最佳模型

## 注册 Gymnasium 环境

在 `__init__.py` 中注册，支持 `gym.make("Go1Balance-v0")` 等调用方式。

## 依赖

- gymnasium >= 1.0.0
- mujoco >= 3.0
- stable-baselines3 >= 2.0
- numpy
