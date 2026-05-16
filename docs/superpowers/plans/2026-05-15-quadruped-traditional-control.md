# 四足机器人传统控制 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复键盘控制 bug，实现按住加速/松开减速的 Trot 步态遥操，集成姿态平衡反馈。

**Architecture:** 重写 KeyState 为按键状态追踪器，主循环每帧做速度渐变，通过 BalanceGait 获取带平衡补偿的目标关节角，PDController 输出力矩。

**Tech Stack:** Python, MuJoCo, NumPy

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `tools/key_state.py` | 重写 | 按键按下/松开状态追踪 |
| `test_gait.py` | 修改 | 仿真主循环：速度渐变 + BalanceGait 集成 |

---

### Task 1: 重写 KeyState — 按键状态追踪

**Files:**
- Modify: `tools/key_state.py`
- Modify: `test_gait.py` (验证集成)

当前 `on_key` 只处理按下事件。GLFW `launch_passive` 的 `key_callback` 接收 `(key, action, mods)` 三个参数，可以区分 press(1) 和 release(0)。

- [ ] **Step 1: 重写 KeyState 类**

```python
"""Thread-safe keyboard state for MuJoCo viewer teleop."""

import threading

# GLFW key codes
GLFW_KEY_UP = 265
GLFW_KEY_DOWN = 264
GLFW_KEY_LEFT = 263
GLFW_KEY_RIGHT = 262
GLFW_KEY_Q = 81
GLFW_KEY_E = 69
GLFW_KEY_R = 82
GLFW_KEY_SPACE = 32

# Movement keys (tracked for hold state)
MOVE_KEYS = {GLFW_KEY_UP, GLFW_KEY_DOWN, GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_Q, GLFW_KEY_E}


class KeyState:
    """Keyboard state tracker with press/release detection.

    Tracks which keys are currently held. Movement keys are checked
    each frame via is_pressed() for velocity ramping.

    Usage:
        keys = KeyState()
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=keys.on_key)

        # In main loop:
        vx_target = 0.0
        if keys.is_pressed(GLFW_KEY_UP):
            vx_target = 1.0
        if keys.is_pressed(GLFW_KEY_DOWN):
            vx_target = -1.0
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pressed: set = set()
        self.reset = False

    def on_key(self, key: int, action: int, mods: int):
        """GLFW key callback. Pass to launch_passive(key_callback=...)."""
        with self._lock:
            if action == 1:  # GLFW_PRESS
                if key == GLFW_KEY_R:
                    self.reset = True
                elif key in MOVE_KEYS:
                    self._pressed.add(key)
                elif key == GLFW_KEY_SPACE:
                    self._pressed.clear()
            elif action == 0:  # GLFW_RELEASE
                self._pressed.discard(key)

    def is_pressed(self, key: int) -> bool:
        """Check if a key is currently held down."""
        with self._lock:
            return key in self._pressed

    def snapshot(self) -> bool:
        """Read and clear the reset flag.

        Returns:
            True if reset was requested since last call.
        """
        with self._lock:
            r = self.reset
            self.reset = False
            return r
```

- [ ] **Step 2: 验证 KeyState 单元测试**

在项目根目录创建临时测试验证 KeyState 行为：

```python
# test_key_state.py (临时验证，可删除)
from tools.key_state import KeyState, GLFW_KEY_UP, GLFW_KEY_DOWN, GLFW_KEY_LEFT

def test_press_release():
    keys = KeyState()
    keys.on_key(GLFW_KEY_UP, 1, 0)  # press
    assert keys.is_pressed(GLFW_KEY_UP) == True
    keys.on_key(GLFW_KEY_UP, 0, 0)  # release
    assert keys.is_pressed(GLFW_KEY_UP) == False

def test_multiple_keys():
    keys = KeyState()
    keys.on_key(GLFW_KEY_UP, 1, 0)
    keys.on_key(GLFW_KEY_LEFT, 1, 0)
    assert keys.is_pressed(GLFW_KEY_UP) == True
    assert keys.is_pressed(GLFW_KEY_LEFT) == True
    keys.on_key(GLFW_KEY_UP, 0, 0)
    assert keys.is_pressed(GLFW_KEY_UP) == False
    assert keys.is_pressed(GLFW_KEY_LEFT) == True

def test_space_clears_all():
    keys = KeyState()
    keys.on_key(GLFW_KEY_UP, 1, 0)
    keys.on_key(GLFW_KEY_LEFT, 1, 0)
    keys.on_key(GLFW_KEY_SPACE, 1, 0)
    assert keys.is_pressed(GLFW_KEY_UP) == False
    assert keys.is_pressed(GLFW_KEY_LEFT) == False

def test_reset_flag():
    keys = KeyState()
    assert keys.snapshot() == False
    keys.on_key(ord('R'), 1, 0)
    assert keys.snapshot() == True
    assert keys.snapshot() == False

if __name__ == "__main__":
    test_press_release()
    test_multiple_keys()
    test_space_clears_all()
    test_reset_flag()
    print("All tests passed!")
```

Run: `python test_key_state.py`
Expected: `All tests passed!`

- [ ] **Step 3: Commit**

```bash
git add tools/key_state.py test_key_state.py
git commit -m "fix: rewrite KeyState with press/release tracking"
```

---

### Task 2: 修改 test_gait.py — 速度渐变 + BalanceGait 集成

**Files:**
- Modify: `test_gait.py`

核心改动：
1. 用 `is_pressed()` 查询按键状态，每帧计算目标速度
2. 速度渐变：`vx += (vx_target - vx) * alpha`，`alpha = 1 - exp(-dt/tau)`
3. 集成 `BalanceGait` 替代直接调用 `TrotGait`

- [ ] **Step 1: 重写 test_gait.py**

```python
"""Trot gait teleop with velocity ramping and balance feedback.

Controls:
  UP/DOWN   - forward/backward (hold to accelerate)
  LEFT/RIGHT - turn (hold to rotate)
  Q/E        - lateral left/right
  SPACE      - emergency stop
  R          - reset simulation
  ESC        - quit
"""

import mujoco
import mujoco.viewer
import time
import numpy as np

from tools.balance_gait import BalanceGait
from tools.pd_controller import PDController
from tools.key_state import KeyState, GLFW_KEY_UP, GLFW_KEY_DOWN, \
    GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_Q, GLFW_KEY_E

# Speed limits
VX_MAX = 1.5        # m/s forward
VX_MIN = -0.8       # m/s backward
YAW_MAX = 1.5       # rad/s turn rate
VY_MAX = 0.5        # m/s lateral

# Velocity ramping time constant (seconds)
TAU = 0.3


def get_body_state(data):
    """Extract body height, pitch, roll from MuJoCo data."""
    body_z = data.qpos[2]
    quat = data.qpos[3:7]  # wxyz

    w, x, y, z = quat
    pitch = np.arcsin(np.clip(2 * (w * y - x * z), -1, 1))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    ang_vel_x = data.qvel[3]
    ang_vel_y = data.qvel[4]

    return body_z, pitch, roll, ang_vel_x, ang_vel_y


def ramp(current, target, alpha):
    """Exponential smoothing toward target."""
    return current + (target - current) * alpha


def main():
    model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene.xml")
    data = mujoco.MjData(model)

    ctrl = PDController(n_joints=12, kp=100.0, kd=1.5)
    gait = BalanceGait(freq=3.0, target_height=0.27)
    keys = KeyState()

    # Current velocities (ramped)
    vx = 0.0
    vy = 0.0
    yaw = 0.0

    print("Controls: UP/DOWN=fwd/back  LEFT/RIGHT=turn  Q/E=lateral  SPACE=stop  R=reset")

    with mujoco.viewer.launch_passive(model, data, key_callback=keys.on_key) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            # --- Reset ---
            if keys.snapshot():
                mujoco.mj_resetDataKeyframe(model, data, 0)
                mujoco.mj_forward(model, data)
                vx = vy = yaw = 0.0
                continue

            # --- Compute target velocity from key state ---
            vx_target = 0.0
            yaw_target = 0.0
            vy_target = 0.0

            if keys.is_pressed(GLFW_KEY_UP):
                vx_target = VX_MAX
            if keys.is_pressed(GLFW_KEY_DOWN):
                vx_target = VX_MIN
            if keys.is_pressed(GLFW_KEY_LEFT):
                yaw_target = YAW_MAX
            if keys.is_pressed(GLFW_KEY_RIGHT):
                yaw_target = -YAW_MAX
            if keys.is_pressed(GLFW_KEY_Q):
                vy_target = VY_MAX
            if keys.is_pressed(GLFW_KEY_E):
                vy_target = -VY_MAX

            # --- Velocity ramping ---
            alpha = 1.0 - np.exp(-dt / TAU)
            vx = ramp(vx, vx_target, alpha)
            vy = ramp(vy, vy_target, alpha)
            yaw = ramp(yaw, yaw_target, alpha)

            # Snap to zero when very small (avoid drift)
            if abs(vx) < 0.01:
                vx = 0.0
            if abs(vy) < 0.01:
                vy = 0.0
            if abs(yaw) < 0.01:
                yaw = 0.0

            walking = abs(vx) > 0.01 or abs(vy) > 0.01 or abs(yaw) > 0.01

            # --- Body state ---
            body_z, pitch, roll, ang_vel_x, ang_vel_y = get_body_state(data)

            # --- BalanceGait: gait + balance compensation ---
            ref = gait.compute(
                t=data.time,
                vx=vx, vy=vy, yaw_rate=yaw,
                body_z=body_z,
                pitch=pitch, roll=roll,
                ang_vel_x=ang_vel_x, ang_vel_y=ang_vel_y,
                walking=walking,
                dt=dt,
            )

            # --- PD control ---
            ctrl.apply(ref, data)

            # --- Step simulation ---
            mujoco.mj_step(model, data)

            # --- Camera follow ---
            alpha_cam = 0.1
            viewer.cam.lookat[:] = (1 - alpha_cam) * viewer.cam.lookat[:] + alpha_cam * data.qpos[:3]
            viewer.sync()

            # --- Status ---
            status = "[RECOVER]" if gait.is_recovering else ""
            print(f"\r{status} vx={vx:+.2f} vy={vy:+.2f} yaw={yaw:+.2f}  "
                  f"pitch={np.degrees(pitch):+.1f}° roll={np.degrees(roll):+.1f}°", end="")

            # --- Real-time sync ---
            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行仿真验证**

```bash
python test_gait.py
```

验证项：
- 按住 UP 键，机器狗逐渐加速前进
- 松开 UP 键，机器狗逐渐减速停下
- 按住 LEFT/RIGHT，持续转向
- 松开方向键，转向停止
- SPACE 急停
- R 重置
- 姿态过大时显示 [RECOVER]

- [ ] **Step 3: Commit**

```bash
git add test_gait.py
git commit -m "feat: velocity ramping + BalanceGait integration in teleop"
```

---

### Task 3: 清理临时测试文件

- [ ] **Step 1: 删除临时测试**

```bash
rm test_key_state.py
```

- [ ] **Step 2: 最终 Commit**

```bash
git add -A
git commit -m "chore: cleanup temporary test file"
```
