"""Keyboard teleoperation for Go2 in MuJoCo viewer with balance control.

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
import numpy as np

from tools.balance_gait import BalanceGait
from tools.pd_controller import PDController
from tools.key_state import KeyState

# Home pose for Go2
HIP_BASE = 0.0
THIGH_BASE = 0.8
CALF_BASE = -1.5
HOME_QPOS = [HIP_BASE, THIGH_BASE, CALF_BASE] * 4


def get_body_state(data):
    """Extract body height, pitch, roll from MuJoCo data."""
    body_z = data.qpos[2]
    w, x, y, z = data.qpos[3:7]
    pitch = np.arcsin(np.clip(2 * (w * y - x * z), -1, 1))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return body_z, pitch, roll, data.qvel[3], data.qvel[4]


def main():
    model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene.xml")
    data = mujoco.MjData(model)

    # Init home pose
    data.qpos[:7] = [0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0]
    for leg in range(4):
        for j in range(3):
            data.qpos[7 + leg * 3 + j] = HOME_QPOS[leg * 3 + j]
    mujoco.mj_forward(model, data)

    ctrl = PDController(n_joints=12, kp=50.0, kd=1.5)
    gait = BalanceGait(freq=4.0, target_height=0.35)
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

            body_z, pitch, roll, ang_vel_x, ang_vel_y = get_body_state(data)

            ref = gait.compute(
                t=data.time,
                vx=vx, vy=vy, yaw_rate=yaw,
                body_z=body_z,
                pitch=pitch, roll=roll,
                ang_vel_x=ang_vel_x, ang_vel_y=ang_vel_y,
                walking=walking,
                dt=model.opt.timestep,
            )

            ctrl.apply(ref, data)

            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
