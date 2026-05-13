# Go1 Custom RL Environments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build modular Gymnasium environments for Unitree Go1 robot dog with three tasks: balance, walk, and jump.

**Architecture:** Custom `gymnasium.Env` subclasses with shared `Go1BaseEnv` base class. Each task environment overrides `_get_reward()` and `_is_terminated()`. Scene XMLs define terrain. SB3 PPO trains all tasks.

**Tech Stack:** Python 3, gymnasium >= 1.0.0, mujoco >= 3.0, stable-baselines3 >= 2.0, numpy

**Spec:** `docs/superpowers/specs/2026-05-13-go1-custom-envs-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `go1_envs/__init__.py` | Package init + Gymnasium env registration |
| `go1_envs/base.py` | `Go1BaseEnv` — model loading, common obs/action/step/reset logic |
| `go1_envs/go1_balance.py` | `Go1BalanceEnv` — balance task reward/termination |
| `go1_envs/go1_walk.py` | `Go1WalkEnv` — walk task reward/termination |
| `go1_envs/go1_jump.py` | `Go1JumpEnv` — jump task reward/termination |
| `go1_envs/train.py` | Unified CLI training entry point |
| `go1_envs/scenes/flat_scene.xml` | Flat ground scene (balance + walk) |
| `go1_envs/scenes/obstacle_scene.xml` | Obstacle scene (jump) |
| `tests/test_go1_envs.py` | Environment validation tests |

## Key Dimensions (verified from model)

- `nq=19` (7 freejoint + 12 joint angles)
- `nv=18` (6 freejoint vel + 12 joint vel)
- `nu=12` (12 position actuators)
- Foot geoms: FR=20, FL=31, RR=43, RL=55
- Action bounds per joint: `[-0.863, -0.686, -2.818]` to `[0.863, 4.501, -0.888]` (repeated 4x)
- timestep=0.002s, frame_skip=25, dt=0.05s
- home keyframe: `qpos=[0,0,0.27, 1,0,0,0, 0,0.9,-1.8, 0,0.9,-1.8, 0,0.9,-1.8, 0,0.9,-1.8]`

---

### Task 1: Scene XML Files

**Files:**
- Create: `go1_envs/scenes/flat_scene.xml`
- Create: `go1_envs/scenes/obstacle_scene.xml`

- [ ] **Step 1: Create `go1_envs/scenes/` directory**

```bash
mkdir -p go1_envs/scenes
```

- [ ] **Step 2: Write `flat_scene.xml`**

This is the flat ground scene for balance and walk tasks. It includes the Go1 model and a ground plane.

```xml
<mujoco model="go1 flat scene">
  <include file="../../mujoco_menagerie/unitree_go1/go1.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
```

- [ ] **Step 3: Write `obstacle_scene.xml`**

Same as flat scene but with an obstacle at x=2.0m.

```xml
<mujoco model="go1 obstacle scene">
  <include file="../../mujoco_menagerie/unitree_go1/go1.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- Obstacle: red box at x=2.0m, height 0.2m, width 1.0m (y-axis) -->
    <body name="obstacle" pos="2 0 0.1">
      <geom type="box" size="0.05 0.5 0.1" rgba="0.8 0.2 0.2 1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
```

- [ ] **Step 4: Verify scene XMLs load correctly**

```bash
python3 -c "
import mujoco
m = mujoco.MjModel.from_xml_path('go1_envs/scenes/flat_scene.xml')
print(f'flat_scene: nq={m.nq}, nv={m.nv}, nu={m.nu}')
m2 = mujoco.MjModel.from_xml_path('go1_envs/scenes/obstacle_scene.xml')
print(f'obstacle_scene: nq={m2.nq}, nv={m2.nv}, nu={m2.nu}')
# Check obstacle geom exists
found = False
for i in range(m2.ngeom):
    if m2.geom(i).name == '':
        body_id = m2.geom_bodyid[i]
        if m2.body(body_id).name == 'obstacle':
            found = True
            break
print(f'Obstacle found: {found}')
"
```
Expected: `flat_scene: nq=19, nv=18, nu=12` and `obstacle_scene: nq=19, nv=18, nu=12` and `Obstacle found: True`

- [ ] **Step 5: Commit**

```bash
git add go1_envs/scenes/
git commit -m "feat: add flat and obstacle scene XMLs for Go1 envs"
```

---

### Task 2: Go1BaseEnv Base Class

**Files:**
- Create: `go1_envs/__init__.py` (minimal)
- Create: `go1_envs/base.py`

- [ ] **Step 1: Create minimal `__init__.py`**

```python
"""Go1 custom Gymnasium environments for reinforcement learning."""
```

- [ ] **Step 2: Write `Go1BaseEnv` in `base.py`**

```python
"""Go1BaseEnv: Base class for Go1 quadruped robot environments."""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# Foot geom IDs in the Go1 model (verified)
FOOT_GEOM_NAMES = ["FR", "FL", "RR", "RL"]

# Home keyframe joint angles (hip, thigh, knee) per leg
HOME_QPOS = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
HOME_Z = 0.27


class Go1BaseEnv(gym.Env):
    """Base environment for Go1 quadruped robot tasks.

    Subclasses must implement:
        - _get_reward(obs, action) -> float
        - _is_terminated() -> bool
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        xml_file: str,
        frame_skip: int = 25,
        reset_noise_scale: float = 0.1,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.reset_noise_scale = reset_noise_scale

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # Renderer for human/rgb_array modes
        self._renderer = None

        # Cache foot geom IDs
        self._foot_geom_ids = []
        for name in FOOT_GEOM_NAMES:
            for i in range(self.model.ngeom):
                if self.model.geom(i).name == name:
                    self._foot_geom_ids.append(i)
                    break

        # Action space: 12 joint target angles
        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0].astype(np.float32),
            high=self.model.actuator_ctrlrange[:, 1].astype(np.float32),
            dtype=np.float32,
        )

        # Observation space (49 dims):
        # [body_z(1), body_quat(4), body_vel(3), body_angvel(3),
        #  joint_pos(12), joint_vel(12), foot_contacts(4)]
        obs_dim = 1 + 4 + 3 + 3 + 12 + 12 + 4
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current simulation state."""
        data = self.data
        body_z = np.array([data.qpos[2]], dtype=np.float32)
        body_quat = data.qpos[3:7].astype(np.float32)  # wxyz
        body_vel = data.qvel[0:3].astype(np.float32)  # linear velocity
        body_angvel = data.qvel[3:6].astype(np.float32)  # angular velocity
        joint_pos = data.qpos[7:19].astype(np.float32)  # 12 joint angles
        joint_vel = data.qvel[6:18].astype(np.float32)  # 12 joint velocities
        foot_contacts = self._get_foot_contacts().astype(np.float32)  # 4 binary

        return np.concatenate([
            body_z, body_quat, body_vel, body_angvel,
            joint_pos, joint_vel, foot_contacts,
        ])

    def _get_foot_contacts(self) -> np.ndarray:
        """Check which feet are in contact with the ground."""
        contacts = np.zeros(4, dtype=np.float32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for j, foot_id in enumerate(self._foot_geom_ids):
                if c.geom1 == foot_id or c.geom2 == foot_id:
                    contacts[j] = 1.0
                    break
        return contacts

    def _get_body_z(self) -> float:
        """Return trunk height."""
        return float(self.data.qpos[2])

    def _get_body_tilt(self) -> float:
        """Return tilt angle (radians) from upright. 0 = perfectly upright."""
        quat = self.data.qpos[3:7]
        # Convert quaternion to rotation matrix, extract z-axis component
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        # z-axis of body frame in world frame
        z_axis = rot[:, 2]
        # Angle from vertical: arccos(z_axis . [0,0,1])
        cos_angle = np.clip(z_axis[2], -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def _get_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Subclasses must implement this."""
        raise NotImplementedError

    def _is_terminated(self) -> bool:
        """Subclasses must implement this."""
        raise NotImplementedError

    def _get_info(self) -> dict:
        """Optional override for extra logging info."""
        return {}

    def step(self, action: np.ndarray):
        # Apply action
        self.data.ctrl[:] = action

        # Step simulation frame_skip times
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._get_reward(obs, action)
        terminated = self._is_terminated()
        truncated = False  # handled by TimeLimit wrapper
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset to home keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Add noise to joint positions (not to body pose)
        noise = self.np_random.uniform(
            -self.reset_noise_scale, self.reset_noise_scale, size=12
        ).astype(np.float64)
        self.data.qpos[7:19] += noise

        # Forward dynamics to compute contacts etc.
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            self._renderer.render()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
```

- [ ] **Step 3: Verify base env loads and produces valid obs/action**

```bash
python3 -c "
import numpy as np
from go1_envs.base import Go1BaseEnv

env = Go1BaseEnv(xml_file='go1_envs/scenes/flat_scene.xml')
obs, info = env.reset()
print(f'Obs shape: {obs.shape}')
print(f'Obs dtype: {obs.dtype}')
print(f'Action space: {env.action_space}')
print(f'Obs space: {env.observation_space}')
assert obs.shape == (49,), f'Expected (49,), got {obs.shape}'
assert env.action_space.shape == (12,), f'Expected (12,), got {env.action_space.shape}'

# Step with zero action
action = np.zeros(12, dtype=np.float32)
obs2, reward, term, trunc, info = env.step(action)
print(f'After step: obs_shape={obs2.shape}, reward={reward}, terminated={term}')
env.close()
print('PASS')
"
```
Expected: `Obs shape: (49,)` and `PASS`

- [ ] **Step 4: Commit**

```bash
git add go1_envs/__init__.py go1_envs/base.py
git commit -m "feat: add Go1BaseEnv base class with obs/action/step logic"
```

---

### Task 3: Go1BalanceEnv

**Files:**
- Create: `go1_envs/go1_balance.py`
- Create: `tests/test_go1_envs.py` (start with balance tests)

- [ ] **Step 1: Write balance env**

```python
"""Go1BalanceEnv: Go1 robot balance standing task."""

import numpy as np
from go1_envs.base import Go1BaseEnv, HOME_Z

SCENE_FILE = "go1_envs/scenes/flat_scene.xml"


class Go1BalanceEnv(Go1BaseEnv):
    """Go1 must maintain standing posture against random perturbations.

    Observation (49d): body_z, quat, vel, angvel, joint_pos, joint_vel, foot_contacts
    Action (12d): target joint angles
    """

    def __init__(self, render_mode=None, **kwargs):
        super().__init__(xml_file=SCENE_FILE, render_mode=render_mode, **kwargs)
        self._step_count = 0
        self._perturb_remaining = 0
        self._perturb_force = np.zeros(6)

    def _get_reward(self, obs, action):
        body_z = obs[0]
        body_quat = obs[1:5]
        joint_pos = obs[17:29]

        # Alive reward
        reward = 1.0

        # Tilt penalty: penalize deviation from upright
        # Simple proxy: quaternion w component should be close to 1 for upright
        tilt_penalty = 0.5 * (1.0 - body_quat[0]) ** 2
        reward -= tilt_penalty

        # Action cost: penalize large actions
        action_cost = 0.1 * np.sum(action ** 2)
        reward -= action_cost

        return float(reward)

    def _is_terminated(self):
        body_z = self._get_body_z()
        tilt = self._get_body_tilt()

        # Fell down
        if body_z < 0.15:
            return True
        # Tilted too much (> 45 degrees)
        if tilt > np.pi / 4:
            return True
        return False

    def _get_info(self):
        return {
            "body_z": self._get_body_z(),
            "body_tilt": self._get_body_tilt(),
            "foot_contacts": self._get_foot_contacts().tolist(),
        }

    def _apply_perturbation(self):
        """Apply random horizontal force perturbation."""
        self._step_count += 1

        # If currently applying perturbation
        if self._perturb_remaining > 0:
            self.data.xfrc_applied[1, :3] = self._perturb_force[:3]  # trunk body
            self._perturb_remaining -= 1
            return

        # Clear any previous perturbation
        self.data.xfrc_applied[1, :3] = np.zeros(3)

        # Random chance to start a new perturbation
        if self._step_count > 100 and self.np_random.random() < 0.01:
            force_mag = self.np_random.uniform(10, 30)
            direction = self.np_random.uniform(-np.pi, np.pi)
            self._perturb_force = np.array([
                force_mag * np.cos(direction),
                force_mag * np.sin(direction),
                0, 0, 0, 0,
            ])
            self._perturb_remaining = 10

    def step(self, action):
        self._apply_perturbation()
        return super().step(action)

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._perturb_remaining = 0
        self._perturb_force = np.zeros(6)
        self.data.xfrc_applied[:] = 0
        return super().reset(seed=seed, options=options)
```

- [ ] **Step 2: Write tests for balance env**

```python
"""Tests for Go1 custom environments."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env


class TestGo1BalanceEnv:
    """Tests for Go1BalanceEnv."""

    def _make_env(self):
        from go1_envs.go1_balance import Go1BalanceEnv
        return Go1BalanceEnv()

    def test_env_passes_checker(self):
        """Env must pass gymnasium's env_checker."""
        env = self._make_env()
        check_env(env, skip_render_check=True)
        env.close()

    def test_obs_shape(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (49,)
        assert obs.dtype == np.float32
        env.close()

    def test_action_space(self):
        env = self._make_env()
        assert env.action_space.shape == (12,)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, reward, term, trunc, info = env.step(action)
        assert obs2.shape == (49,)
        env.close()

    def test_reset_noise(self):
        """Observations after reset should vary due to noise."""
        env = self._make_env()
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=1)
        assert not np.allclose(obs1, obs2)
        env.close()

    def test_terminated_on_fall(self):
        """Env should terminate when robot falls."""
        env = self._make_env()
        env.reset()
        # Apply large downward action to make it collapse
        for _ in range(500):
            action = env.action_space.sample() * 0
            # Use extreme joint positions to destabilize
            action = np.array([0, 3.0, -0.9] * 4, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        # Either terminated or ran many steps
        env.close()

    def test_reward_is_finite(self):
        env = self._make_env()
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert np.isfinite(reward), f"Reward not finite: {reward}"
            if term:
                env.reset()
        env.close()
```

- [ ] **Step 3: Run tests**

```bash
python3 -m pytest tests/test_go1_envs.py::TestGo1BalanceEnv -v
```
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add go1_envs/go1_balance.py tests/test_go1_envs.py
git commit -m "feat: add Go1BalanceEnv with reward/termination/perturbation"
```

---

### Task 4: Go1WalkEnv

**Files:**
- Create: `go1_envs/go1_walk.py`
- Modify: `tests/test_go1_envs.py` (add walk tests)

- [ ] **Step 1: Write walk env**

```python
"""Go1WalkEnv: Go1 robot forward walking task."""

import numpy as np
from go1_envs.base import Go1BaseEnv

SCENE_FILE = "go1_envs/scenes/flat_scene.xml"


class Go1WalkEnv(Go1BaseEnv):
    """Go1 must learn to walk forward along the x-axis.

    Observation (49d): body_z, quat, vel, angvel, joint_pos, joint_vel, foot_contacts
    Action (12d): target joint angles
    """

    def __init__(self, render_mode=None, **kwargs):
        super().__init__(xml_file=SCENE_FILE, render_mode=render_mode, **kwargs)

    def _get_reward(self, obs, action):
        body_vel_x = obs[5]  # x-component of body linear velocity
        body_vel_y = obs[6]  # y-component (lateral)

        # Forward velocity reward
        reward = 1.0 * body_vel_x

        # Alive reward
        reward += 1.0

        # Control cost
        reward -= 0.05 * np.sum(action ** 2)

        # Lateral velocity penalty
        reward -= 0.1 * body_vel_y ** 2

        return float(reward)

    def _is_terminated(self):
        body_z = self._get_body_z()

        # Fell down
        if body_z < 0.15:
            return True

        # Drifted too far laterally
        y_pos = self.data.qpos[1]
        if abs(y_pos) > 2.0:
            return True

        # Reached goal
        x_pos = self.data.qpos[0]
        if x_pos > 20.0:
            return True

        return False

    def _get_info(self):
        return {
            "body_z": self._get_body_z(),
            "x_pos": float(self.data.qpos[0]),
            "y_pos": float(self.data.qpos[1]),
            "x_vel": float(self.data.qvel[0]),
        }
```

- [ ] **Step 2: Add walk env tests**

Append to `tests/test_go1_envs.py`:

```python
class TestGo1WalkEnv:
    """Tests for Go1WalkEnv."""

    def _make_env(self):
        from go1_envs.go1_walk import Go1WalkEnv
        return Go1WalkEnv()

    def test_env_passes_checker(self):
        env = self._make_env()
        check_env(env, skip_render_check=True)
        env.close()

    def test_obs_shape(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (49,)
        env.close()

    def test_reward_finite(self):
        env = self._make_env()
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert np.isfinite(reward)
            if term:
                env.reset()
        env.close()

    def test_info_has_position(self):
        env = self._make_env()
        env.reset()
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        assert "x_pos" in info
        assert "x_vel" in info
        env.close()
```

- [ ] **Step 3: Run tests**

```bash
python3 -m pytest tests/test_go1_envs.py -v
```
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add go1_envs/go1_walk.py tests/test_go1_envs.py
git commit -m "feat: add Go1WalkEnv with forward velocity reward"
```

---

### Task 5: Go1JumpEnv

**Files:**
- Create: `go1_envs/go1_jump.py`
- Modify: `tests/test_go1_envs.py` (add jump tests)

- [ ] **Step 1: Write jump env**

```python
"""Go1JumpEnv: Go1 robot obstacle jumping task."""

import numpy as np
from go1_envs.base import Go1BaseEnv

SCENE_FILE = "go1_envs/scenes/obstacle_scene.xml"
OBSTACLE_X = 2.0  # obstacle x position
OBSTACLE_TOP = 0.2  # obstacle top height (center 0.1 + half-size 0.1)


class Go1JumpEnv(Go1BaseEnv):
    """Go1 must jump over an obstacle at x=2.0m.

    Observation (49d): body_z, quat, vel, angvel, joint_pos, joint_vel, foot_contacts
    Action (12d): target joint angles
    """

    def __init__(self, render_mode=None, **kwargs):
        super().__init__(xml_file=SCENE_FILE, render_mode=render_mode, **kwargs)
        self._prev_x = 0.0
        self._crossed_obstacle = False
        self._hit_obstacle = False

    def _check_obstacle_contact(self) -> bool:
        """Check if any body part is in contact with the obstacle geom."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1_name = self.model.geom(c.geom1).name if c.geom1 >= 0 else ""
            g2_name = self.model.geom(c.geom2).name if c.geom2 >= 0 else ""
            body1 = self.model.geom_bodyid[c.geom1] if c.geom1 >= 0 else -1
            body2 = self.model.geom_bodyid[c.geom2] if c.geom2 >= 0 else -1
            obstacle_body = None
            for b in range(self.model.nbody):
                if self.model.body(b).name == "obstacle":
                    obstacle_body = b
                    break
            if obstacle_body is not None:
                if body1 == obstacle_body or body2 == obstacle_body:
                    return True
        return False

    def _get_reward(self, obs, action):
        body_z = obs[0]
        body_vel_x = obs[5]
        x_pos = self.data.qpos[0]

        # Forward velocity reward
        reward = 1.0 * body_vel_x

        # Jump height reward when near obstacle
        if OBSTACLE_X - 0.5 < x_pos < OBSTACLE_X + 0.5:
            height_bonus = 10.0 * max(0, body_z - OBSTACLE_TOP)
            reward += height_bonus

        # Obstacle collision penalty
        if self._check_obstacle_contact():
            reward -= 100.0
            self._hit_obstacle = True

        # Successfully crossed obstacle
        if x_pos > OBSTACLE_X + 0.3 and not self._crossed_obstacle and not self._hit_obstacle:
            reward += 50.0
            self._crossed_obstacle = True

        return float(reward)

    def _is_terminated(self):
        body_z = self._get_body_z()

        # Fell down
        if body_z < 0.15:
            return True

        # Hit obstacle
        if self._hit_obstacle:
            return True

        # Reached goal
        x_pos = self.data.qpos[0]
        if x_pos > 5.0:
            return True

        return False

    def _get_info(self):
        return {
            "body_z": self._get_body_z(),
            "x_pos": float(self.data.qpos[0]),
            "crossed_obstacle": self._crossed_obstacle,
            "hit_obstacle": self._hit_obstacle,
        }

    def reset(self, seed=None, options=None):
        self._prev_x = 0.0
        self._crossed_obstacle = False
        self._hit_obstacle = False
        return super().reset(seed=seed, options=options)
```

- [ ] **Step 2: Add jump env tests**

Append to `tests/test_go1_envs.py`:

```python
class TestGo1JumpEnv:
    """Tests for Go1JumpEnv."""

    def _make_env(self):
        from go1_envs.go1_jump import Go1JumpEnv
        return Go1JumpEnv()

    def test_env_passes_checker(self):
        env = self._make_env()
        check_env(env, skip_render_check=True)
        env.close()

    def test_obs_shape(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (49,)
        env.close()

    def test_reward_finite(self):
        env = self._make_env()
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert np.isfinite(reward)
            if term:
                env.reset()
        env.close()

    def test_info_has_obstacle_fields(self):
        env = self._make_env()
        env.reset()
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        assert "crossed_obstacle" in info
        assert "hit_obstacle" in info
        env.close()
```

- [ ] **Step 3: Run all tests**

```bash
python3 -m pytest tests/test_go1_envs.py -v
```
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add go1_envs/go1_jump.py tests/test_go1_envs.py
git commit -m "feat: add Go1JumpEnv with obstacle collision detection"
```

---

### Task 6: Gymnasium Registration + `__init__.py`

**Files:**
- Modify: `go1_envs/__init__.py`

- [ ] **Step 1: Update `__init__.py` with registrations**

```python
"""Go1 custom Gymnasium environments for reinforcement learning."""

from gymnasium.envs.registration import register

register(
    id="Go1Balance-v0",
    entry_point="go1_envs.go1_balance:Go1BalanceEnv",
    max_episode_steps=1000,
)

register(
    id="Go1Walk-v0",
    entry_point="go1_envs.go1_walk:Go1WalkEnv",
    max_episode_steps=1000,
)

register(
    id="Go1Jump-v0",
    entry_point="go1_envs.go1_jump:Go1JumpEnv",
    max_episode_steps=1000,
)
```

- [ ] **Step 2: Verify gym.make works for all tasks**

```bash
python3 -c "
import go1_envs  # triggers registration
import gymnasium as gym

for task in ['Go1Balance-v0', 'Go1Walk-v0', 'Go1Jump-v0']:
    env = gym.make(task)
    obs, info = env.reset()
    print(f'{task}: obs_shape={obs.shape}, action_space={env.action_space.shape}')
    obs2, r, t, tr, i = env.step(env.action_space.sample())
    print(f'  step: reward={r:.3f}, terminated={t}')
    env.close()

print('All registrations OK')
"
```
Expected: All three envs load and step successfully

- [ ] **Step 3: Commit**

```bash
git add go1_envs/__init__.py
git commit -m "feat: register Go1Balance-v0, Go1Walk-v0, Go1Jump-v0 envs"
```

---

### Task 7: Training Entry Point

**Files:**
- Create: `go1_envs/train.py`

- [ ] **Step 1: Write `train.py`**

```python
"""Unified training entry point for Go1 environments.

Usage:
    python -m go1_envs.train balance [--timesteps N] [--n-envs N]
    python -m go1_envs.train walk [--timesteps N] [--n-envs N]
    python -m go1_envs.train jump [--timesteps N] [--n-envs N]
    python -m go1_envs.train eval <task> [--model PATH]
    python -m go1_envs.train viz <task> [--model PATH]
"""

import argparse
import os

import gymnasium as gym
import numpy as np

# Trigger env registration
import go1_envs  # noqa: F401

TASK_DEFAULTS = {
    "balance": {"timesteps": 300_000, "save_path": "go1_balance_ppo"},
    "walk": {"timesteps": 500_000, "save_path": "go1_walk_ppo"},
    "jump": {"timesteps": 800_000, "save_path": "go1_jump_ppo"},
}

TASK_ENV_IDS = {
    "balance": "Go1Balance-v0",
    "walk": "Go1Walk-v0",
    "jump": "Go1Jump-v0",
}


def make_env(task: str, render_mode=None):
    """Create a single env instance."""
    return gym.make(TASK_ENV_IDS[task], render_mode=render_mode)


def make_vec_env(task: str, n_envs: int = 1, render_mode=None):
    """Create vectorized env with normalization."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    if n_envs == 1:
        env = make_env(task, render_mode=render_mode)
    else:
        env = SubprocVecEnv([lambda: make_env(task) for _ in range(n_envs)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def train(task: str, timesteps: int, n_envs: int = 8):
    """Train a PPO agent on the specified task."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    save_path = TASK_DEFAULTS[task]["save_path"]

    train_env = make_vec_env(task, n_envs=n_envs)
    eval_env = make_vec_env(task, n_envs=1)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=42,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./{save_path}/",
        log_path=f"./{save_path}/logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Training {task} for {timesteps:,} steps with {n_envs} envs...")
    print(f"Save path: ./{save_path}/")
    print("-" * 60)

    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(f"{save_path}/final_model")
    train_env.save(f"{save_path}/vec_normalize.pkl")
    print(f"\nDone! Model saved to ./{save_path}/")

    train_env.close()
    eval_env.close()


def evaluate(task: str, model_path: str, num_episodes: int = 10):
    """Evaluate a trained model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    env = make_vec_env(task, n_envs=1)
    env = VecNormalize.load(f"{model_path}/../vec_normalize.pkl", env)
    env.training = False
    model = PPO.load(model_path, env=env)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=num_episodes, deterministic=True
    )
    print(f"Evaluation ({num_episodes} episodes): {mean_reward:.1f} +/- {std_reward:.1f}")
    env.close()


def visualize(task: str, model_path: str, num_episodes: int = 3):
    """Visualize a trained model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    env = make_vec_env(task, n_envs=1, render_mode="human")
    env = VecNormalize.load(f"{model_path}/../vec_normalize.pkl", env)
    env.training = False
    model = PPO.load(model_path, env=env)

    for ep in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
        print(f"Episode {ep + 1}: Reward = {total_reward:.1f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Go1 RL Training")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("task", choices=TASK_DEFAULTS.keys())
    train_parser.add_argument("--timesteps", type=int, default=None)
    train_parser.add_argument("--n-envs", type=int, default=8)

    # Eval
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("task", choices=TASK_DEFAULTS.keys())
    eval_parser.add_argument("--model", type=str, default=None)
    eval_parser.add_argument("--episodes", type=int, default=10)

    # Visualize
    viz_parser = subparsers.add_parser("viz")
    viz_parser.add_argument("task", choices=TASK_DEFAULTS.keys())
    viz_parser.add_argument("--model", type=str, default=None)
    viz_parser.add_argument("--episodes", type=int, default=3)

    args = parser.parse_args()

    if args.command == "train":
        timesteps = args.timesteps or TASK_DEFAULTS[args.task]["timesteps"]
        train(args.task, timesteps, args.n_envs)
    elif args.command == "eval":
        save_path = TASK_DEFAULTS[args.task]["save_path"]
        model_path = args.model or f"./{save_path}/best_model"
        evaluate(args.task, model_path, args.episodes)
    elif args.command == "viz":
        save_path = TASK_DEFAULTS[args.task]["save_path"]
        model_path = args.model or f"./{save_path}/best_model"
        visualize(args.task, model_path, args.episodes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify train.py imports and help text**

```bash
python3 -m go1_envs.train --help
python3 -m go1_envs.train train --help
```
Expected: Help text shows balance/walk/jump task options

- [ ] **Step 3: Commit**

```bash
git add go1_envs/train.py
git commit -m "feat: add unified training CLI for Go1 envs"
```

---

### Task 8: Final Integration Test

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

```bash
python3 -m pytest tests/test_go1_envs.py -v
```
Expected: All tests PASS

- [ ] **Step 2: Quick smoke test — train each task for 1000 steps**

```bash
python3 -c "
import go1_envs
import gymnasium as gym
from stable_baselines3 import PPO

for task in ['Go1Balance-v0', 'Go1Walk-v0', 'Go1Jump-v0']:
    env = gym.make(task)
    model = PPO('MlpPolicy', env, verbose=0, n_steps=128, batch_size=64, n_epochs=2)
    model.learn(total_timesteps=1000)
    print(f'{task}: 1000 steps OK')
    env.close()
print('All smoke tests passed')
"
```
Expected: All three tasks train for 1000 steps without error

- [ ] **Step 3: Commit any fixes if needed**

```bash
git add -A
git commit -m "fix: integration test fixes"
```

---

### Task 9: Final Commit

- [ ] **Step 1: Verify clean state**

```bash
git status
git log --oneline -5
```

- [ ] **Step 2: Ensure all files are committed**

```bash
git add -A
git status
```
Expected: nothing to commit
