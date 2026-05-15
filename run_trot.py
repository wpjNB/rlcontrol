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

# GLFW key codes
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

VX_MAX = 1.5
VX_MIN = -0.8
YAW_MAX = 1.5
VY_MAX = 0.5
TAU = 0.3


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
    quat = data.qpos[3:7]
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

    print("Go2 Trot Gait - IK-based Traditional Control")
    print("Controls: UP/DOWN=fwd/back  LEFT/RIGHT=turn  Q/E=lateral  SPACE=stop  R=reset")

    with mujoco.viewer.launch_passive(model, data, key_callback=keys.on_key) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            if keys.snapshot():
                mujoco.mj_resetDataKeyframe(model, data, 0)
                mujoco.mj_forward(model, data)
                vx = vy = yaw = 0.0
                continue

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

            body_z, pitch, roll, ang_vel_x, ang_vel_y = get_body_state(data)

            ref = controller.compute(
                t=data.time,
                vx=vx, vy=vy, yaw_rate=yaw,
                body_z=body_z,
                pitch=pitch, roll=roll,
                ang_vel_x=ang_vel_x, ang_vel_y=ang_vel_y,
                walking=walking,
                dt=dt,
            )

            n_joints = 12
            qpos = data.qpos[7:7 + n_joints]
            qvel = data.qvel[6:6 + n_joints]
            data.ctrl[:n_joints] = kp * (ref - qpos) - kd * qvel

            mujoco.mj_step(model, data)

            alpha_cam = 0.1
            viewer.cam.lookat[:] = (1 - alpha_cam) * viewer.cam.lookat[:] + alpha_cam * data.qpos[:3]
            viewer.sync()

            status = "[RECOVER]" if controller.is_recovering else ""
            print(f"\r{status} vx={vx:+.2f} vy={vy:+.2f} yaw={yaw:+.2f}  "
                  f"pitch={np.degrees(pitch):+.1f} roll={np.degrees(roll):+.1f} z={body_z:.3f}", end="")

            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
