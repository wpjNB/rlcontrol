"""Keyboard teleoperation for Go2 in MuJoCo viewer.

Controls:
  W/S     - forward/backward
  A/D     - turn left/right
  Q/E     - lateral left/right
  P       - toggle walk/pause
  SPACE   - stop and stand still
  R       - reset episode
  ESC     - quit
"""

import mujoco
import mujoco.viewer
import time
import math

from tools.pd_controller import PDController
from tools.key_state import KeyState

# Home pose
HIP_BASE = 0.0
THIGH_BASE = 0.8
CALF_BASE = -1.5
HOME_QPOS = [HIP_BASE, THIGH_BASE, CALF_BASE] * 4

# Gait parameters
FREQ = 4.0
THIGH_AMP = 0.4
CALF_AMP = 0.25
HIP_AMP = 0.15
TROT_PHASES = [0.0, math.pi, math.pi, 0.0]


def main():
    model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene.xml")
    data = mujoco.MjData(model)

    # Init home pose
    data.qpos[:7] = [0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0]
    for leg in range(4):
        for j in range(3):
            data.qpos[7 + leg * 3 + j] = HOME_QPOS[leg * 3 + j]
    mujoco.mj_forward(model, data)

    ctrl_standing = PDController(n_joints=12, kp=80.0, kd=3.0)
    ctrl_walking = PDController(n_joints=12, kp=50.0, kd=1.5)
    keys = KeyState()

    print("Controls: W/S=fwd/back  A/D=turn  Q/E=lateral  P=walk/pause  SPACE=stop  R=reset")

    with mujoco.viewer.launch_passive(model, data, key_callback=keys.on_key) as viewer:
        while viewer.is_running():
            step_start = time.time()
            vx, vy, yaw, walking, do_reset = keys.snapshot()

            if do_reset:
                data.qpos[:7] = [0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0]
                for leg in range(4):
                    for j in range(3):
                        data.qpos[7 + leg * 3 + j] = HOME_QPOS[leg * 3 + j]
                data.qvel[:] = 0
                data.ctrl[:] = 0
                mujoco.mj_forward(model, data)
                continue

            if walking:
                t = data.time
                omega = 2.0 * math.pi * FREQ
                fwd_offset = vx * 0.15
                ref = [0.0] * 12

                for leg in range(4):
                    phase = TROT_PHASES[leg]
                    base = leg * 3

                    # Hip: yaw + lateral
                    hip = yaw * HIP_AMP
                    if leg in [0, 2]:
                        ref[base] = hip - vy * 0.1
                    else:
                        ref[base] = -hip + vy * 0.1

                    # Thigh: oscillation + forward bias
                    ref[base + 1] = THIGH_BASE + fwd_offset + THIGH_AMP * math.sin(omega * t + phase)
                    # Calf: cos (90° offset)
                    ref[base + 2] = CALF_BASE + CALF_AMP * math.cos(omega * t + phase)

                ctrl_walking.apply(ref, data)
            else:
                ctrl_standing.apply(HOME_QPOS, data)

            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
