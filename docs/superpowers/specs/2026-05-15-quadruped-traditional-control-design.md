# 四足机器人传统控制 — Trot 步态 + 键盘遥操设计

## 目标

通过传统控制（正弦轨迹 + PD 力矩）实现 Go1/Go2 四足机器人的 Trot 步态行走，支持键盘按住加速/松开减速的实时遥操，集成姿态平衡反馈。

## 现有基础

| 模块 | 文件 | 功能 |
|------|------|------|
| `TrotGait` | `go1_envs/gait.py` | 正弦轨迹生成，支持 vx/vy/yaw_rate |
| `BalanceGait` | `tools/balance_gait.py` | 姿态补偿 + 跌倒恢复 |
| `PDController` | `tools/pd_controller.py` | 关节 PD 力矩控制 |
| `KeyState` | `tools/key_state.py` | 键盘输入（有 bug） |
| `test_gait.py` | `test_gait.py` | 仿真主循环 |

## 改动范围

只改 2 个文件：`tools/key_state.py` 和 `test_gait.py`。

### 1. 重写 `tools/key_state.py`

**问题**：当前 `on_key` 只处理按下事件，左右转松开后 yaw 不归零，前进后退需要反复按键。

**修复**：

- 跟踪每个键的**按下/松开**状态（利用 GLFW 回调的 `action` 参数）
- 提供 `is_pressed(key)` 查询接口
- `snapshot()` 返回当前所有按键状态，供主循环做速度渐变

```python
class KeyState:
    def __init__(self):
        self._pressed = set()  # 当前被按住的键集合
        self.reset = False

    def on_key(self, key, action):
        # action: GLFW_PRESS=1, GLFW_RELEASE=0
        if action == GLFW_PRESS:
            self._pressed.add(key)
        elif action == GLFW_RELEASE:
            self._pressed.discard(key)

    def is_pressed(self, key) -> bool:
        return key in self._pressed

    def snapshot(self):
        """返回当前按键状态快照 + reset 标志"""
        ...
```

### 2. 修改 `test_gait.py`

**主循环速度控制逻辑**：

每帧根据按键状态做速度渐变：

```
vx 目标值：
  UP 按住 → vx_target = +vx_max
  DOWN 按住 → vx_target = -vx_max
  都没按 → vx_target = 0

vx 当前值渐变：
  vx += (vx_target - vx) * alpha   # alpha = 1 - exp(-dt / tau)
  tau = 0.3s（加速时间常数）

yaw 目标值：
  LEFT 按住 → yaw_target = +yaw_max
  RIGHT 按住 → yaw_target = -yaw_max
  都没按 → yaw_target = 0
```

**集成 BalanceGait**：

- 每帧从 MuJoCo data 提取 body_z, pitch, roll, ang_vel
- 传入 `BalanceGait.compute()` 获取带平衡补偿的目标关节角
- `PDController.apply()` 输出力矩

**键盘映射**：

| 键 | 功能 |
|----|------|
| UP | 前进（按住加速） |
| DOWN | 后退（按住加速） |
| LEFT | 左转（按住持续转向） |
| RIGHT | 右转（按住持续转向） |
| Q | 左侧移 |
| E | 右侧移 |
| SPACE | 急停（速度归零） |
| R | 重置仿真 |

## 架构

```
KeyState (按键状态)
    ↓ is_pressed()
主循环 (速度渐变)
    ↓ vx, vy, yaw_rate
BalanceGait.compute() (正弦轨迹 + 姿态补偿)
    ↓ 12d 目标关节角
PDController.apply() (PD 力矩)
    ↓ 力矩
MuJoCo mj_step()
```

## 不做的事

- 不新增文件
- 不加 Walk/Pace/Gallop 步态
- 不加速度档位切换
- 不改 BalanceGait / PDController / TrotGait
