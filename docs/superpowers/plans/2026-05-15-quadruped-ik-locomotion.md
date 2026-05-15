# Go2 IK-Based Trot Locomotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete traditional control system for Go2 quadruped trot locomotion using Cartesian foot trajectories + inverse kinematics, with keyboard teleop, velocity ramping, balance compensation, and fall recovery.

**Architecture:** 5-layer pipeline: KeyState → Velocity Ramping → GaitPhaseScheduler (per-leg swing/stance) → FootTrajectory + IK (Cartesian → joint angles) → PD torque control. All new code in `traditional_control/` directory, no modifications to existing files.

**Tech Stack:** Python 3.10, MuJoCo, NumPy. Run with: `/home/wpj/miniconda3/envs/rlcontrol/bin/python`

---

## File Structure

| File | Create/Modify | Responsibility |
|------|--------------|----------------|
| `traditional_control/__init__.py` | Create | Package init |
| `traditional_control/kinematics.py` | Create | Go2 kinematic constants, FK, IK, joint clamping |
| `traditional_control/gait_phase.py` | Create | Trot phase scheduler, step length calculation |
| `traditional_control/foot_trajectory.py` | Create | Cartesian foot trajectory generation |
| `traditional_control/trot_controller.py` | Create | Main controller: phase + trajectory + IK + balance + recovery |
| `run_trot.py` | Create | MuJoCo viewer main loop: KeyState + velocity ramp + PD control |
| `tests/test_traditional_control.py` | Create | Unit tests for all modules |

---

### Task 1: kinematics.py — Go2 运动学

**Files:**
- Create: `traditional_control/__init__.py`
- Create: `traditional_control/kinematics.py`
- Create: `tests/test_traditional_control.py`

- [ ] **Step 1: Create package init**

```python
# traditional_control/__init__.py
```

- [ ] **Step 2: Write IK tests**

```python
# tests/test_traditional_control.py
import numpy as np
import sys
sys.path.insert(0, '.')

from traditional_control.kinematics import Go2LegKinematics, HOME_QPOS


def test_fk_ik_roundtrip():
    """FK(IK(pos)) should recover the original position."""
    kin = Go2LegKinematics()
    # Target foot position relative to hip (reasonable standing config)
    target = np.array([0.0, 0.0, -0.4])
    joints = kin.solve('FL', target)
    recovered = kin.forward('FL', joints)
    np.testing.assert_allclose(recovered, target, atol=1e-4)


def test_ik_standing_pose():
    """IK for a straight-down foot should give thigh~0, calf~negative."""
    kin = Go2LegKinematics()
    # Foot directly below hip, z = -(L1+L2) = -0.426
    target = np.array([0.0, 0.0, -(kin.L1 + kin.L2)])
    joints = kin.solve('FL', target)
    # hip should be ~0 (straight down)
    assert abs(joints[0]) < 0.01
    # thigh should be ~0
    assert abs(joints[1]) < 0.01
    # calf should be ~0 (fully extended)
    assert abs(joints[2]) < 0.01


def test_ik_bent_knee():
    """IK for a closer foot should give negative calf angle (bent)."""
    kin = Go2LegKinematics()
    target = np.array([0.0, 0.0, -0.3])  # closer than full extension
    joints = kin.solve('FL', target)
    assert joints[2] < 0  # calf negative = bent


def test_clamp_joints():
    """clamp_joints should enforce joint limits."""
    kin = Go2LegKinematics()
    joints = np.array([5.0, 5.0, -5.0])  # way out of range
    clamped = kin.clamp_joints('FL', joints)
    limits = kin.get_limits('FL')
    for i in range(3):
        assert limits[i][0] <= clamped[i] <= limits[i][1]


def test_fk_standing():
    """FK of HOME_QPOS should give a reasonable foot position."""
    kin = Go2LegKinematics()
    # HOME_QPOS: [hip, thigh, knee] per leg, standing pose
    # For FL: joints = HOME_QPOS[0:3]
    joints = HOME_QPOS[0:3]
    pos = kin.forward('FL', joints)
    # Foot should be below hip (z < 0)
    assert pos[2] < 0
    # Foot should be roughly at standing height
    assert -0.5 < pos[2] < -0.2


if __name__ == '__main__':
    test_fk_ik_roundtrip()
    test_ik_standing_pose()
    test_ik_bent_knee()
    test_clamp_joints()
    test_fk_standing()
    print("All kinematics tests passed!")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/home/wpj/miniconda3/envs/rlcontrol/bin/python tests/test_traditional_control.py`
Expected: `ModuleNotFoundError: No module named 'traditional_control'`

- [ ] **Step 4: Implement kinematics.py**

```python
# traditional_control/kinematics.py
"""Go2 quadruped leg kinematics — FK, IK, joint limits."""

import numpy as np

# Go2 kinematic parameters (extracted from MuJoCo model)
HIP_OFFSET = {
    'FL': np.array([0.1934,  0.0465, 0.0]),
    'FR': np.array([0.1934, -0.0465, 0.0]),
    'RL': np.array([-0.1934,  0.0465, 0.0]),
    'RR': np.array([-0.1934, -0.0465, 0.0]),
}

HIP_LINK = 0.0955   # hip abduction link length (y direction)
L1 = 0.213          # thigh length (z direction)
L2 = 0.213          # calf length (z direction)

# Joint limits from MuJoCo model
JOINT_LIMITS = {
    'FL': [(-1.047, 1.047), (-1.571, 3.491), (-2.723, -0.838)],
    'FR': [(-1.047, 1.047), (-1.571, 3.491), (-2.723, -0.838)],
    'RL': [(-1.047, 1.047), (-0.524, 4.538), (-2.723, -0.838)],
    'RR': [(-1.047, 1.047), (-0.524, 4.538), (-2.723, -0.838)],
}

# Standing pose joint angles
HOME_QPOS = np.array([
    0.0, 0.9, -1.8,   # FL: hip, thigh, knee
    0.0, 0.9, -1.8,   # FR
    0.0, 0.9, -1.8,   # RL
    0.0, 0.9, -1.8,   # RR
])

LEGS = ['FL', 'FR', 'RL', 'RR']


class Go2LegKinematics:
    """3-DOF leg kinematics for Unitree Go2."""

    def __init__(self):
        self.L1 = L1
        self.L2 = L2
        self.hip_link = HIP_LINK

    def forward(self, leg: str, joints: np.ndarray) -> np.ndarray:
        """Forward kinematics: joint angles → foot position relative to hip.

        Args:
            leg: 'FL', 'FR', 'RL', or 'RR'
            joints: [hip, thigh, calf] angles in radians

        Returns:
            [x, y, z] foot position relative to hip joint
        """
        q_hip, q_thigh, q_calf = joints

        # Hip abduction: y offset
        y = self.hip_link * np.cos(q_hip) - self.hip_link

        # Planar FK (thigh + calf in the x-z plane)
        # Thigh endpoint
        x1 = self.L1 * np.sin(q_thigh)
        z1 = -self.L1 * np.cos(q_thigh)

        # Calf endpoint relative to thigh
        x2 = self.L2 * np.sin(q_thigh + q_calf)
        z2 = -self.L2 * np.cos(q_thigh + q_calf)

        x = x1 + x2
        z = z1 + z2

        # Apply hip rotation to x component
        x_eff = x * np.cos(q_hip)

        return np.array([x_eff, y, z])

    def solve(self, leg: str, foot_pos: np.ndarray) -> np.ndarray:
        """Inverse kinematics: foot position → joint angles.

        Args:
            leg: 'FL', 'FR', 'RL', or 'RR'
            foot_pos: [x, y, z] foot position relative to hip joint

        Returns:
            [hip, thigh, calf] joint angles in radians
        """
        x, y, z = foot_pos

        # Hip abduction angle
        # y = hip_link * cos(q_hip) - hip_link
        # cos(q_hip) = (y + hip_link) / hip_link
        cos_hip = np.clip((y + self.hip_link) / self.hip_link, -1.0, 1.0)
        q_hip = np.arccos(cos_hip)
        # Sign: positive y = abduction outward
        if y < 0:
            q_hip = -q_hip

        # Project into the sagittal plane
        # After hip rotation, the effective x is reduced
        x_eff = x / max(np.cos(q_hip), 0.1)  # avoid division by zero

        # 2-link IK in the sagittal plane (x_eff, z)
        d_sq = x_eff**2 + z**2
        d = np.sqrt(d_sq)

        # Cosine rule for calf angle
        cos_calf = (d_sq - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_calf = np.clip(cos_calf, -1.0, 1.0)
        q_calf = -np.arccos(cos_calf)  # negative = bent

        # Thigh angle
        alpha = np.arctan2(x_eff, -z)
        beta = np.arctan2(self.L2 * np.sin(q_calf),
                          self.L1 + self.L2 * np.cos(q_calf))
        q_thigh = alpha - beta

        return np.array([q_hip, q_thigh, q_calf])

    def clamp_joints(self, leg: str, joints: np.ndarray) -> np.ndarray:
        """Clamp joint angles to hardware limits."""
        limits = JOINT_LIMITS[leg]
        result = np.copy(joints)
        for i in range(3):
            result[i] = np.clip(result[i], limits[i][0], limits[i][1])
        return result

    def get_limits(self, leg: str):
        """Return joint limits for a leg."""
        return JOINT_LIMITS[leg]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/home/wpj/miniconda3/envs/rlcontrol/bin/python tests/test_traditional_control.py`
Expected: `All kinematics tests passed!`

- [ ] **Step 6: Commit**

```bash
git add traditional_control/__init__.py traditional_control/kinematics.py tests/test_traditional_control.py
git commit -m "feat: add Go2 leg kinematics with FK/IK and joint limits"
```

---

### Task 2: gait_phase.py — 步态相位调度

**Files:**
- Create: `traditional_control/gait_phase.py`
- Modify: `tests/test_traditional_control.py`

- [ ] **Step 1: Write phase scheduler tests**

Append to `tests/test_traditional_control.py`:

```python
from traditional_control.gait_phase import GaitPhaseScheduler


def test_phase_trot_diagonal_sync():
    """Trot: FR/RL in phase, FL/RR in phase, offset by pi."""
    sched = GaitPhaseScheduler(freq=2.0)
    phases = sched.step(0.0, 0.001)
    # At t=0: FR phase=0, FL phase=pi
    assert abs(phases['FR']['phase'] - 0.0) < 0.01
    assert abs(phases['FL']['phase'] - np.pi) < 0.01
    assert abs(phases['RL']['phase'] - np.pi) < 0.01
    assert abs(phases['RR']['phase'] - 0.0) < 0.01


def test_phase_swing_stance():
    """Phase should correctly identify swing vs stance."""
    sched = GaitPhaseScheduler(freq=2.0, duty_cycle=0.6)
    # At t=0, FR phase=0 → swing (swing is 0 to 2pi*(1-duty))
    phases = sched.step(0.0, 0.001)
    assert phases['FR']['is_swing'] == True  # phase=0 < swing_end
    # At t = 1/(2*freq) = 0.25s, FR phase = pi → stance
    phases = sched.step(0.25, 0.001)
    assert phases['FR']['is_swing'] == False  # phase=pi > swing_end


def test_step_length_forward():
    """Step length should scale with vx."""
    sched = GaitPhaseScheduler(freq=2.0)
    step = sched.get_step_length(vx=1.0, yaw_rate=0.0, leg='FL')
    assert abs(step - 0.5) < 0.01  # 1.0 / 2.0 = 0.5


def test_step_length_turning():
    """Turning should create differential step lengths."""
    sched = GaitPhaseScheduler(freq=2.0)
    left = sched.get_step_length(vx=0.0, yaw_rate=1.0, leg='FL')
    right = sched.get_step_length(vx=0.0, yaw_rate=1.0, leg='FR')
    assert left > 0   # left leg steps forward
    assert right < 0   # right leg steps backward
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/wpj/miniconda3/envs/rlcontrol/bin/python tests/test_traditional_control.py`
Expected: `ModuleNotFoundError: No module named 'traditional_control.gait_phase'`

- [ ] **Step 3: Implement gait_phase.py**

```python
# traditional_control/gait_phase.py
"""Trot gait phase scheduler with step length calculation."""

import numpy as np

# Trot phase offsets: FR/RL in sync, FL/RR in sync, offset by pi
TROT_OFFSETS = {
    'FL': np.pi,
    'FR': 0.0,
    'RL': np.pi,
    'RR': 0.0,
}

LEGS = ['FL', 'FR', 'RL', 'RR']
LEFT_LEGS = {'FL', 'RL'}
RIGHT_LEGS = {'FR', 'RR'}


class GaitPhaseScheduler:
    """Manages swing/stance phase timing for each leg in a trot gait.

    Args:
        freq: Gait frequency in Hz (default 2.0)
        duty_cycle: Fraction of cycle spent in stance (default 0.6)
    """

    def __init__(self, freq: float = 2.0, duty_cycle: float = 0.6):
        self.freq = freq
        self.duty_cycle = duty_cycle
        self.swing_end = 2 * np.pi * (1 - duty_cycle)  # swing phase ends here

    def step(self, t: float, dt: float) -> dict:
        """Compute phase info for all legs at time t.

        Returns:
            Dict mapping leg name -> {
                'phase': raw phase 0~2pi,
                'is_swing': True if in swing phase,
                'phase_norm': normalized progress within current phase (0~1)
            }
        """
        omega = 2 * np.pi * self.freq
        result = {}

        for leg in LEGS:
            phase = (omega * t + TROT_OFFSETS[leg]) % (2 * np.pi)
            is_swing = phase < self.swing_end

            if is_swing:
                phase_norm = phase / self.swing_end
            else:
                phase_norm = (phase - self.swing_end) / (2 * np.pi - self.swing_end)

            result[leg] = {
                'phase': phase,
                'is_swing': is_swing,
                'phase_norm': phase_norm,
            }

        return result

    def get_step_length(self, vx: float, yaw_rate: float, leg: str) -> float:
        """Calculate step length for a leg based on velocity commands.

        Args:
            vx: Forward velocity (m/s)
            yaw_rate: Yaw rate (rad/s, positive = turn left)
            leg: Leg name

        Returns:
            Step length in meters
        """
        base_step = vx / max(self.freq, 0.1)
        turn_offset = yaw_rate * 0.1 / max(self.freq, 0.1)

        if leg in LEFT_LEGS:
            return base_step + turn_offset
        else:
            return base_step - turn_offset

    def get_lateral_offset(self, vy: float) -> float:
        """Calculate lateral foot offset from vy command."""
        return vy / max(self.freq, 0.1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/wpj/miniconda3/envs/rlcontrol/bin/python tests/test_traditional_control.py`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add traditional_control/gait_phase.py tests/test_traditional_control.py
git commit -m "feat: add trot gait phase scheduler with step length calculation"
```

---

### Task 3: foot_trajectory.py — 足端轨迹生成

**Files:**
- Create: `traditional_control/foot_trajectory.py`
- Modify: `tests/test_traditional_control.py`

- [ ] **Step 1: Write trajectory tests**

Append to `tests/test_traditional_control.py`:

```python
from traditional_control.foot_trajectory import FootTrajectory


def test_swing_trajectory_peak():
    """Swing at midpoint (phase_norm=0.5) should have max height."""
    traj = FootTrajectory(step_height=0.06)
    pos = traj.compute(phase_norm=0.5, is_swing=True, step_len=0.1, vy_offset=0.0)
    assert abs(pos[2] - 0.06) < 0.001  # peak height


def test_swing_trajectory_start_end():
    """Swing at start/end should be at ground level."""
    traj = FootTrajectory(step_height=0.06)
    pos_start = traj.compute(phase_norm=0.0, is_swing=True, step_len=0.1, vy_offset=0.0)
    pos_end = traj.compute(phase_norm=1.0, is_swing=True, step_len=0.1, vy_offset=0.0)
    assert abs(pos_start[2]) < 0.001
    assert abs(pos_end[2]) < 0.001


def test_stance_trajectory_ground():
    """Stance phase should keep foot on ground."""
    traj = FootTrajectory()
    pos = traj.compute(phase_norm=0.5, is_swing=False, step_len=0.1, vy_offset=0.0)
    assert abs(pos[2]) < 0.001  # on ground


def test_step_length_clamping():
    """Step length should be clamped to max."""
    traj = FootTrajectory(step_length_max=0.15)
    pos = traj.compute(phase_norm=0.5, is_swing=True, step_len=1.0, vy_offset=0.0)
    # Step should be clamped to 0.15, so x range is [-0.075, 0.075]
    assert abs(pos[0]) <= 0.076


def test_lateral_offset():
    """vy_offset should appear in y component."""
    traj = FootTrajectory()
    pos = traj.compute(phase_norm=0.5, is_swing=True, step_len=0.0, vy_offset=0.05)
    assert abs(pos[1] - 0.05) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/wpj/miniconda3/envs/rlcontrol/bin/python tests/test_traditional_control.py`
Expected: `ModuleNotFoundError: No module named 'traditional_control.foot_trajectory'`

- [ ] **Step 3: Implement foot_trajectory.py**

```python
# traditional_control/foot_trajectory.py
"""Cartesian foot trajectory generation for trot gait."""

import numpy as np


class FootTrajectory:
    """Generates foot trajectories in Cartesian space.

    Swing phase: sinusoidal lift + linear forward sweep.
    Stance phase: foot planted, body moves forward over it.

    Args:
        step_height: Maximum foot lift height in meters (default 0.06)
        step_length_max: Maximum step length in meters (default 0.2)
    """

    def __init__(self, step_height: float = 0.06, step_length_max: float = 0.2):
        self.step_height = step_height
        self.step_length_max = step_length_max

    def compute(
        self,
        phase_norm: float,
        is_swing: bool,
        step_len: float,
        vy_offset: float,
    ) -> np.ndarray:
        """Compute foot target position relative to hip joint.

        Args:
            phase_norm: Normalized progress within current phase (0~1)
            is_swing: True if in swing phase
            step_len: Desired step length in meters (sign = direction)
            vy_offset: Lateral offset in meters

        Returns:
            [x, y, z] foot position relative to hip
        """
        # Clamp step length
        step_len = np.clip(step_len, -self.step_length_max, self.step_length_max)

        if is_swing:
            # Swing: linear forward sweep + sinusoidal lift
            x = -step_len / 2 + step_len * phase_norm
            z = self.step_height * np.sin(np.pi * phase_norm)
        else:
            # Stance: foot planted, body moves over it
            x = step_len / 2 - step_len * phase_norm
            z = 0.0

        y = vy_offset

        return np.array([x, y, z])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/wpj/miniconda3/envs/rlcontrol/bin/python tests/test_traditional_control.py`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add traditional_control/foot_trajectory.py tests/test_traditional_control.py
git commit -m "feat: add Cartesian foot trajectory generation for trot gait"
```

---

### Task 4: trot_controller.py — 主控制器

**Files:**
- Create: `traditional_control/trot_controller.py`
- Modify: `tests/test_traditional_control.py`

- [ ] **Step 1: Write controller tests**

Append to `tests/test_traditional_control.py`:

```python
from traditional_control.trot_controller import TrotController


def test_standing_output():
    """Standing (walking=False) should return HOME_QPOS."""
    ctrl = TrotController()
    ref = ctrl.compute(t=0.0, vx=0.0, vy=0.0, yaw_rate=0.0,
                       body_z=0.27, pitch=0.0, roll=0.0,
                       ang_vel_x=0.0, ang_vel_y=0.0, walking=False, dt=0.002)
    np.testing.assert_allclose(ref, HOME_QPOS, atol=1e-6)


def test_walking_output_shape():
    """Walking output should be 12d array."""
    ctrl = TrotController()
    ref = ctrl.compute(t=0.0, vx=0.5, vy=0.0, yaw_rate=0.0,
                       body_z=0.27, pitch=0.0, roll=0.0,
                       ang_vel_x=0.0, ang_vel_y=0.0, walking=True, dt=0.002)
    assert ref.shape == (12,)


def test_walking_differs_from_standing():
    """Walking output should differ from HOME_QPOS."""
    ctrl = TrotController()
    ref = ctrl.compute(t=0.1, vx=0.5, vy=0.0, yaw_rate=0.0,
                       body_z=0.27, pitch=0.0, roll=0.0,
                       ang_vel_x=0.0, ang_vel_y=0.0, walking=True, dt=0.002)
    assert not np.allclose(ref, HOME_QPOS, atol=0.01)


def test_fall_recovery():
    """Large tilt should trigger recovery pose."""
    ctrl = TrotController()
    ref = ctrl.compute(t=0.0, vx=0.5, vy=0.0, yaw_rate=0.0,
                       body_z=0.27, pitch=0.5, roll=0.0,  # > 0.35 tilt
                       ang_vel_x=0.0, ang_vel_y=0.0, walking=True, dt=0.002)
    # Recovery pose should have more flexed knees (more negative calf)
    for leg in range(4):
        assert ref[leg * 3 + 2] < HOME_QPOS[leg * 3 + 2] + 0.01


def test_turn_differential():
    """Turning should create different hip angles for left vs right legs."""
    ctrl = TrotController()
    ref = ctrl.compute(t=0.1, vx=0.0, vy=0.0, yaw_rate=1.0,
                       body_z=0.27, pitch=0.0, roll=0.0,
                       ang_vel_x=0.0, ang_vel_y=0.0, walking=True, dt=0.002)
    # Left legs (FL=0, RL=6) and right legs (FR=3, RR=9) should differ
    # At least one pair should have different hip angles
    left_hip_avg = (ref[0] + ref[6]) / 2
    right_hip_avg = (ref[3] + ref[9]) / 2
    assert abs(left_hip_avg - right_hip_avg) > 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/wpj/miniconda3/envs/rlcontrol/bin/python tests/test_traditional_control.py`
Expected: `ModuleNotFoundError: No module named 'traditional_control.trot_controller'`

- [ ] **Step 3: Implement trot_controller.py**

```python
# traditional_control/trot_controller.py
"""Main trot controller: combines phase scheduling, trajectory, IK, and balance."""

import numpy as np
from traditional_control.kinematics import Go2LegKinematics, HOME_QPOS, LEGS
from traditional_control.gait_phase import GaitPhaseScheduler
from traditional_control.foot_trajectory import FootTrajectory

# Leg indices in the 12d joint array
LEG_INDICES = {'FL': 0, 'FR': 3, 'RL': 6, 'RR': 9}
FRONT_LEGS = ['FL', 'FR']
REAR_LEGS = ['RL', 'RR']
LEFT_LEGS = ['FL', 'RL']   # body left
RIGHT_LEGS = ['FR', 'RR']  # body right


class TrotController:
    """Trot gait controller with IK-based trajectory and balance compensation.

    Args:
        freq: Gait frequency Hz (default 2.0)
        duty_cycle: Stance phase fraction (default 0.6)
        step_height: Foot lift height m (default 0.06)
        target_height: Target body height m (default 0.27)
        tilt_limit: Max tilt before recovery rad (default 0.35)
    """

    def __init__(
        self,
        freq: float = 2.0,
        duty_cycle: float = 0.6,
        step_height: float = 0.06,
        target_height: float = 0.27,
        tilt_limit: float = 0.35,
    ):
        self.kin = Go2LegKinematics()
        self.phase = GaitPhaseScheduler(freq=freq, duty_cycle=duty_cycle)
        self.trajectory = FootTrajectory(step_height=step_height)
        self.target_height = target_height
        self.tilt_limit = tilt_limit

        # Balance compensation gains
        self.kp_pitch = 1.5
        self.kp_roll = 1.0
        self.kp_height = 5.0

        # Fall recovery state
        self._recovering = False
        self._recover_timer = 0.0
        self._recover_duration = 0.5

    def compute(
        self,
        t: float,
        vx: float = 0.0,
        vy: float = 0.0,
        yaw_rate: float = 0.0,
        body_z: float = 0.27,
        pitch: float = 0.0,
        roll: float = 0.0,
        ang_vel_x: float = 0.0,
        ang_vel_y: float = 0.0,
        walking: bool = True,
        dt: float = 0.002,
    ) -> np.ndarray:
        """Compute 12d reference joint angles.

        Args:
            t: Current time in seconds
            vx, vy, yaw_rate: Velocity commands
            body_z: Body height in meters
            pitch, roll: Body attitude in radians
            ang_vel_x, ang_vel_y: Angular velocity in rad/s
            walking: Whether gait is active
            dt: Timestep for recovery timer

        Returns:
            12d numpy array of target joint angles [hip,thigh,calf] x [FL,FR,RL,RR]
        """
        # --- Fall detection ---
        tilt = np.sqrt(pitch**2 + roll**2)
        ang_vel_mag = np.sqrt(ang_vel_x**2 + ang_vel_y**2)

        if not self._recovering and (
            tilt > self.tilt_limit or (tilt > 0.2 and ang_vel_mag > 2.0)
        ):
            self._recovering = True
            self._recover_timer = 0.0

        # --- Recovery mode ---
        if self._recovering:
            self._recover_timer += dt
            ref = self._recovery_pose(pitch, roll)
            if tilt < 0.1 and self._recover_timer > self._recover_duration:
                self._recovering = False
            return ref

        # --- Standing mode ---
        if not walking:
            return HOME_QPOS.copy()

        # --- Walking mode: per-leg trajectory + IK ---
        phases = self.phase.step(t, dt)
        lateral_offset = self.phase.get_lateral_offset(vy)
        ref = np.zeros(12)

        for leg in LEGS:
            idx = LEG_INDICES[leg]
            p = phases[leg]
            step_len = self.phase.get_step_length(vx, yaw_rate, leg)

            # Foot trajectory in Cartesian space
            foot_pos = self.trajectory.compute(
                phase_norm=p['phase_norm'],
                is_swing=p['is_swing'],
                step_len=step_len,
                vy_offset=lateral_offset,
            )

            # Inverse kinematics
            joints = self.kin.solve(leg, foot_pos)
            joints = self.kin.clamp_joints(leg, joints)

            ref[idx:idx + 3] = joints

        # --- Balance compensation ---
        ref = self._apply_balance(ref, pitch, roll, body_z, ang_vel_x, ang_vel_y)

        return ref

    def _apply_balance(
        self,
        ref: np.ndarray,
        pitch: float,
        roll: float,
        body_z: float,
        ang_vel_x: float,
        ang_vel_y: float,
    ) -> np.ndarray:
        """PD-based attitude and height compensation."""
        # Pitch compensation
        pitch_corr = self.kp_pitch * pitch + 0.3 * ang_vel_y
        for leg in FRONT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= pitch_corr * 0.3  # thigh
            ref[idx + 2] += pitch_corr * 0.2  # knee
        for leg in REAR_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += pitch_corr * 0.3
            ref[idx + 2] -= pitch_corr * 0.2

        # Roll compensation
        roll_corr = self.kp_roll * roll + 0.2 * ang_vel_x
        for leg in RIGHT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= roll_corr * 0.2
            ref[idx + 0] -= roll_corr * 0.1
        for leg in LEFT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += roll_corr * 0.2
            ref[idx + 0] += roll_corr * 0.1

        # Height compensation
        height_err = self.target_height - body_z
        height_corr = self.kp_height * height_err
        for leg in LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= height_corr * 0.2
            ref[idx + 2] += height_corr * 0.15

        return ref

    def _recovery_pose(self, pitch: float, roll: float) -> np.ndarray:
        """Crouch pose to recover from near-fall."""
        ref = HOME_QPOS.copy()

        # Crouch: flex knees more
        for leg in LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 2] += 0.3  # more knee flex

        # Counter-tilt
        pitch_corr = 0.8 * pitch
        for leg in FRONT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= pitch_corr * 0.4
        for leg in REAR_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += pitch_corr * 0.4

        roll_corr = 0.6 * roll
        for leg in RIGHT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] -= roll_corr * 0.3
        for leg in LEFT_LEGS:
            idx = LEG_INDICES[leg]
            ref[idx + 1] += roll_corr * 0.3

        return ref

    @property
    def is_recovering(self) -> bool:
        return self._recovering
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/wpj/miniconda3/envs/rlcontrol/bin/python tests/test_traditional_control.py`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add traditional_control/trot_controller.py tests/test_traditional_control.py
git commit -m "feat: add main trot controller with IK trajectory and balance compensation"
```

---

### Task 5: run_trot.py — MuJoCo 仿真主循环

**Files:**
- Create: `run_trot.py`

- [ ] **Step 1: Implement run_trot.py**

```python
# run_trot.py
"""Go2 trot gait visualization with keyboard teleop.

Controls:
  UP/W     - forward (hold to accelerate)
  DOWN/S   - backward (hold to accelerate)
  LEFT/A   - turn left (hold)
  RIGHT/D  - turn right (hold)
  Q        - lateral left
  E        - lateral right
  SPACE    - emergency stop
  R        - reset simulation
  ESC      - quit
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
import threading

from traditional_control.trot_controller import TrotController

# --- GLFW key codes ---
GLFW_KEY_SPACE = 32
GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT = 263
GLFW_KEY_DOWN = 264
GLFW_KEY_UP = 265
GLFW_KEY_A = 65
GLFW_KEY_D = 68
GLFW_KEY_E = 69
GLFW_KEY_Q = 81
GLFW_KEY_R = 82
GLFW_KEY_S = 83
GLFW_KEY_W = 87
GLFW_PRESS = 1
GLFW_RELEASE = 0

# --- Speed limits ---
VX_MAX = 1.5
VX_MIN = -0.8
YAW_MAX = 1.5
VY_MAX = 0.5
TAU = 0.3  # velocity ramping time constant


class KeyState:
    """Minimal keyboard state tracker with press/release."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pressed = set()
        self.reset = False

    def on_key(self, key: int, action: int, mods: int):
        with self._lock:
            if action == GLFW_PRESS:
                if key == GLFW_KEY_R:
                    self.reset = True
                elif key == GLFW_KEY_SPACE:
                    self._pressed.clear()
                else:
                    self._pressed.add(key)
            elif action == GLFW_RELEASE:
                self._pressed.discard(key)

    def is_pressed(self, key: int) -> bool:
        with self._lock:
            return key in self._pressed

    def snapshot(self) -> bool:
        with self._lock:
            r = self.reset
            self.reset = False
            return r


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

    controller = TrotController(freq=2.0, target_height=0.27)
    keys = KeyState()

    kp = 100.0
    kd = 1.5

    vx = 0.0
    vy = 0.0
    yaw = 0.0

    print("Go2 Trot Gait — IK-based Traditional Control")
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

            # --- Target velocity from keys ---
            vx_target = 0.0
            yaw_target = 0.0
            vy_target = 0.0

            if keys.is_pressed(GLFW_KEY_UP) or keys.is_pressed(GLFW_KEY_W):
                vx_target = VX_MAX
            if keys.is_pressed(GLFW_KEY_DOWN) or keys.is_pressed(GLFW_KEY_S):
                vx_target = VX_MIN
            if keys.is_pressed(GLFW_KEY_LEFT) or keys.is_pressed(GLFW_KEY_A):
                yaw_target = YAW_MAX
            if keys.is_pressed(GLFW_KEY_RIGHT) or keys.is_pressed(GLFW_KEY_D):
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

            if abs(vx) < 0.01:
                vx = 0.0
            if abs(vy) < 0.01:
                vy = 0.0
            if abs(yaw) < 0.01:
                yaw = 0.0

            walking = abs(vx) > 0.01 or abs(vy) > 0.01 or abs(yaw) > 0.01

            # --- Body state ---
            body_z, pitch, roll, ang_vel_x, ang_vel_y = get_body_state(data)

            # --- Controller ---
            ref = controller.compute(
                t=data.time,
                vx=vx, vy=vy, yaw_rate=yaw,
                body_z=body_z,
                pitch=pitch, roll=roll,
                ang_vel_x=ang_vel_x, ang_vel_y=ang_vel_y,
                walking=walking,
                dt=dt,
            )

            # --- PD control ---
            n_joints = 12
            qpos = data.qpos[7:7 + n_joints]
            qvel = data.qvel[6:6 + n_joints]
            data.ctrl[:n_joints] = kp * (ref - qpos) - kd * qvel

            # --- Step simulation ---
            mujoco.mj_step(model, data)

            # --- Camera follow ---
            alpha_cam = 0.1
            viewer.cam.lookat[:] = (1 - alpha_cam) * viewer.cam.lookat[:] + alpha_cam * data.qpos[:3]
            viewer.sync()

            # --- Status ---
            status = "[RECOVER]" if controller.is_recovering else ""
            print(f"\r{status} vx={vx:+.2f} vy={vy:+.2f} yaw={yaw:+.2f}  "
                  f"pitch={np.degrees(pitch):+.1f} roll={np.degrees(roll):+.1f} z={body_z:.3f}", end="")

            # --- Real-time sync ---
            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the simulation to verify it works**

Run: `/home/wpj/miniconda3/envs/rlcontrol/bin/python run_trot.py`
Expected: MuJoCo viewer opens. Press UP to accelerate forward, LEFT/RIGHT to turn. Robot should walk with trot gait.

- [ ] **Step 3: Commit**

```bash
git add run_trot.py
git commit -m "feat: add MuJoCo viewer with IK-based trot gait and keyboard teleop"
```

---

### Task 6: Cleanup and final verification

- [ ] **Step 1: Run all tests**

Run: `/home/wpj/miniconda3/envs/rlcontrol/bin/python tests/test_traditional_control.py`
Expected: All tests pass

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "feat: complete IK-based trot locomotion system for Go2"
```
