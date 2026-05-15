"""Keyboard teleoperation for Go2 in MuJoCo viewer.

Controls:
  W/S     - forward/backward
  A/D     - turn left/right
  Q/E     - lateral left/right
  X       - stop all movement (keep walking)
  P       - toggle walk/pause
  R       - reset episode
  ESC     - quit
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
import math
import threading

# GLFW key codes
GLFW_KEY_W = 87
GLFW_KEY_S = 83
GLFW_KEY_A = 65
GLFW_KEY_D = 68
GLFW_KEY_Q = 81
GLFW_KEY_E = 69
GLFW_KEY_X = 88
GLFW_KEY_R = 82
GLFW_KEY_P = 80


class KeyState:
    """Thread-safe keyboard state."""
    def __init__(self):
        self._lock = threading.Lock()
        self.forward = 0.0
        self.lateral = 0.0
        self.turn = 0.0
        self.walking = True  # start walking by default
        self.reset = False

    def on_key(self, key: int):
        with self._lock:
            if key == GLFW_KEY_W:
                self.forward = 1.0
            elif key == GLFW_KEY_S:
                self.forward = -1.0
            elif key == GLFW_KEY_A:
                self.turn = 1.0
            elif key == GLFW_KEY_D:
                self.turn = -1.0
            elif key == GLFW_KEY_Q:
                self.lateral = -1.0
            elif key == GLFW_KEY_E:
                self.lateral = 1.0
            elif key == GLFW_KEY_X:
                self.forward = 0.0
                self.lateral = 0.0
                self.turn = 0.0
            elif key == GLFW_KEY_P:
                self.walking = not self.walking
            elif key == GLFW_KEY_R:
                self.reset = True

    def snapshot(self):
        with self._lock:
            s = (self.forward, self.lateral, self.turn, self.walking, self.reset)
            self.reset = False
            return s


def main():
    model_path = "mujoco_menagerie/unitree_go2/scene.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Home pose
    data.qpos[:7] = np.array([0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0])
    hip_base = 0.0
    thigh_base = 0.8
    calf_base = -1.5
    for leg in range(4):
        data.qpos[7 + leg * 3] = hip_base
        data.qpos[7 + leg * 3 + 1] = thigh_base
        data.qpos[7 + leg * 3 + 2] = calf_base
    mujoco.mj_forward(model, data)

    keys = KeyState()

    # PD params
    kp_stand = 80.0
    kd_stand = 3.0
    kp_walk = 50.0
    kd_walk = 1.5

    freq = 6.0
    thigh_amp = 0.5
    calf_amp = 0.3
    hip_amp = 0.15
    phases = [0.0, math.pi, math.pi, 0.0]  # trot

    print("Controls: W/S=fwd/back  A/D=turn  Q/E=lateral  X=stop  P=pause  R=reset")
    print("Robot starts walking. Press W/A/S/D to steer.")

    with mujoco.viewer.launch_passive(model, data, key_callback=keys.on_key) as viewer:
        while viewer.is_running():
            step_start = time.time()
            fwd, lat, turn, walking, do_reset = keys.snapshot()

            if do_reset:
                data.qpos[:7] = np.array([0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0])
                for leg in range(4):
                    data.qpos[7 + leg * 3] = hip_base
                    data.qpos[7 + leg * 3 + 1] = thigh_base
                    data.qpos[7 + leg * 3 + 2] = calf_base
                data.qvel[:] = 0
                data.ctrl[:] = 0
                mujoco.mj_forward(model, data)
                continue

            t = data.time

            if not walking:
                # Standing: hold home pose
                for leg in range(4):
                    for j in range(3):
                        q_idx = 7 + leg * 3 + j
                        v_idx = 6 + leg * 3 + j
                        target = [hip_base, thigh_base, calf_base][j]
                        torque = kp_stand * (target - data.qpos[q_idx]) - kd_stand * data.qvel[v_idx]
                        data.ctrl[leg * 3 + j] = torque
            else:
                # Walking gait with teleop modulation
                fwd_offset = fwd * 0.3   # forward/backward thigh bias
                for leg in range(4):
                    phase = phases[leg]

                    # Thigh: oscillation + forward bias
                    thigh_angle = thigh_base + fwd_offset + thigh_amp * math.sin(t * freq + phase)
                    # Calf: oscillation
                    calf_angle = calf_base + calf_amp * math.cos(t * freq + phase)
                    # Hip: turn + lateral
                    hip_offset = turn * hip_amp
                    if leg in [0, 2]:  # right legs
                        hip_angle = hip_base + hip_offset - lat * 0.1
                    else:
                        hip_angle = hip_base - hip_offset + lat * 0.1

                    targets = [hip_angle, thigh_angle, calf_angle]

                    for j in range(3):
                        q_idx = 7 + leg * 3 + j
                        v_idx = 6 + leg * 3 + j
                        torque = kp_walk * (targets[j] - data.qpos[q_idx]) - kd_walk * data.qvel[v_idx]
                        data.ctrl[leg * 3 + j] = torque

            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
