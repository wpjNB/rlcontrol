# 四足机器人传统控制 — 笛卡尔空间轨迹 + 逆运动学 Trot 步态

## 目标

用笛卡尔空间足端轨迹 + 逆运动学（IK）实现 Go2 四足机器人 Trot 步态行走，支持键盘遥操（按住加速/松开减速），集成姿态平衡补偿和跌倒恢复。所有代码新建在 `traditional_control/` 目录，不修改现有文件。

## Go2 运动学参数（从 MuJoCo 模型提取）

```
基座高度: 0.445m
髋关节偏移（相对基座中心）:
  FL: (+0.1934, +0.0465, 0)    FR: (+0.1934, -0.0465, 0)
  RL: (-0.1934, +0.0465, 0)    RR: (-0.1934, -0.0465, 0)
髋外展连杆: HIP_LINK = 0.0955m (y 方向，hip→thigh)
大腿长: L1 = 0.213m (z 方向，thigh→calf)
小腿长: L2 = 0.213m (z 方向，calf→foot)
```

**关节限位：**
| 关节 | 前腿范围 | 后腿范围 |
|------|---------|---------|
| hip  | [-1.047, 1.047] | [-1.047, 1.047] |
| thigh | [-1.571, 3.491] | [-0.524, 4.538] |
| calf | [-2.723, -0.838] | [-2.723, -0.838] |

## 文件结构

| 文件 | 职责 |
|------|------|
| `traditional_control/__init__.py` | 包初始化 |
| `traditional_control/kinematics.py` | Go2 运动学常量 + FK/IK |
| `traditional_control/gait_phase.py` | Trot 步态相位调度 + 步长计算 |
| `traditional_control/foot_trajectory.py` | 笛卡尔空间足端轨迹生成 |
| `traditional_control/trot_controller.py` | 主控制器（组合所有层 + 姿态补偿 + 跌倒恢复） |
| `run_trot.py` | MuJoCo 仿真主循环（键盘 + 速度渐变 + PD 控制） |

## 控制架构

```
键盘输入 (KeyState)
    ↓ 按键状态
速度渐变 (exponential ramp, τ=0.3s)
    ↓ vx, vy, yaw_rate
TrotController.compute()
    ├─ 跌倒检测 → recovery_pose（如需要）
    ├─ GaitPhaseScheduler → 每条腿的相位/摆动/支撑
    ├─ 每条腿:
    │   ├─ FootTrajectory → 足端目标 [x, y, z]
    │   └─ LegIK → 3 关节角 [hip, thigh, calf]
    └─ BalanceCompensation → 姿态修正
    ↓ 12d 目标关节角
PD 控制 (torque = kp*(ref-qpos) - kd*qvel)
    ↓ 力矩
MuJoCo mj_step()
```

## 模块设计

### 1. kinematics.py — 运动学

**Go2LegKinematics 类：**
- `__init__()`: 存储连杆参数
- `forward(leg, joints)`: 正运动学 — 关节角 → 足端位置
- `solve(leg, foot_pos)`: 逆运动学 — 足端位置 → 关节角
- `clamp_joints(leg, joints)`: 关节限位裁剪

**逆运动学公式（3-DOF 腿）：**

给定足端位置 `(x, y, z)` 相对髋关节：

```
theta_hip = atan2(y, sqrt(x² + z² - HIP_LINK²))  # 髋外展
y_eff = HIP_LINK - y                               # 有效 y 偏移
d = sqrt(x² + z² - y_eff²)                        # 平面内距离

cos_calf = (d² - L1² - L2²) / (2 * L1 * L2)
theta_calf = -acos(clip(cos_calf, -1, 1))          # 负值 = 弯曲

alpha = atan2(x, -z)                                # 足端方位角
beta = atan2(L2 * sin(theta_calf), L1 + L2 * cos_calf)
theta_thigh = alpha - beta                          # 大腿角
```

### 2. gait_phase.py — 步态相位

**GaitPhaseScheduler 类：**
- `__init__(freq=2.0, duty_cycle=0.6)`: 步频、支撑相比例
- `step(t, dt)`: 返回每条腿的相位信息字典
  - `phase`: 0~2π 原始相位
  - `is_swing`: 是否在摆动相
  - `phase_norm`: 在当前阶段内的归一化进度 (0~1)
- `get_step_length(vx, yaw_rate, leg)`: 根据速度命令计算步长

**Trot 相位偏移：** FR=0, FL=π, RL=π, RR=0

**步长计算：**
```
base_step = vx / freq                    # 基础前后步长
turn_offset = yaw_rate * 0.1 / freq      # 转向差速
左腿步长 = base_step + turn_offset
右腿步长 = base_step - turn_offset
侧移步长 = vy / freq
```

### 3. foot_trajectory.py — 足端轨迹

**FootTrajectory 类：**
- `__init__(step_height=0.06, step_length_max=0.2)`: step_length_max 用于裁剪步长
- `compute(phase_norm, is_swing, step_len, vy_offset)` → `[x, y, z]`
  - `step_len = clip(step_len, -step_length_max, step_length_max)`

**摆动相（0~1 归一化）：**
```
x = -step_len/2 + step_len * phase_norm    # 线性前摆
y = vy_offset                               # 左右偏移
z = step_height * sin(π * phase_norm)       # 半正弦抬腿
```

**支撑相（0~1 归一化）：**
```
x = step_len/2 - step_len * phase_norm      # 足端固定，身体前移
y = vy_offset
z = 0                                       # 贴地
```

### 4. trot_controller.py — 主控制器

**TrotController 类：**
- 组合 GaitPhaseScheduler + FootTrajectory + LegIK
- 姿态平衡补偿（复用 BalanceGait 的补偿逻辑）
- 跌倒检测与恢复

**compute() 主流程：**
1. 跌倒检测：tilt > 0.35 rad → recovery_pose
2. 站立模式：walking=False → HOME_QPOS
3. 步态模式：每条腿 足端轨迹 → IK → 关节角
4. 姿态补偿：pitch/roll/height PD 修正
5. 返回 12d 目标关节角

**姿态补偿（PD 控制）：**
```
pitch_corr = kp_pitch * pitch + 0.3 * ang_vel_y
前腿 thigh -= pitch_corr * 0.3, knee += pitch_corr * 0.2
后腿 thigh += pitch_corr * 0.3, knee -= pitch_corr * 0.2

roll_corr = kp_roll * roll + 0.2 * ang_vel_x
右腿 thigh -= roll_corr * 0.2, hip -= roll_corr * 0.1
左腿 thigh += roll_corr * 0.2, hip += roll_corr * 0.1

height_corr = kp_height * (target_z - body_z)
所有腿 thigh -= height_corr * 0.2, knee += height_corr * 0.15
```

### 5. run_trot.py — 仿真主循环

**自包含入口脚本，不依赖现有 tools/。**

内含简化版 KeyState（按键追踪 + press/release 检测）。

**主循环每帧流程：**
1. 查询按键状态，计算目标速度
2. 速度渐变：`vx += (vx_target - vx) * alpha`，`alpha = 1 - exp(-dt/tau)`
3. `TrotController.compute()` → 12d 目标关节角
4. PD 力矩：`torque = kp * (ref - qpos) - kd * qvel`
5. `mujoco.mj_step()`
6. 相机跟随 + 状态打印

**键盘映射：**
| 键 | 功能 |
|----|------|
| UP / W | 前进（按住加速） |
| DOWN / S | 后退（按住加速） |
| LEFT / A | 左转（按住持续转向） |
| RIGHT / D | 右转（按住持续转向） |
| Q | 左侧移 |
| E | 右侧移 |
| SPACE | 急停 |
| R | 重置仿真 |

## 不做的事

- 不修改现有 `tools/`、`go1_envs/`、`test_gait.py`
- 不加 Walk/Pace/Gallop 多步态
- 不加地形自适应
- 不加视觉/激光雷达感知
