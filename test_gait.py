"""Visualize BalanceGait in MuJoCo viewer.

Controls:
  W/S     - forward/backward
  A/D     - turn left/right
  Q/E     - lateral left/right
  P       - toggle walk/pause
  SPACE   - stop and stand still
  R       - reset episode
  ESC     - quit

Display:
  [RECOVERING] - fall recovery mode active
  pitch/roll   - body attitude in degrees
"""

import mujoco
import mujoco.viewer
import time
import numpy as np

from tools.balance_gait import BalanceGait
from tools.pd_controller import PDController
from tools.key_state import KeyState


def get_body_state(data):
    """Extract body height, pitch, roll from MuJoCo data."""
    body_z = data.qpos[2]  #机体高度
    quat = data.qpos[3:7]  # wxyz

    # Quaternion to euler (pitch, roll)
    w, x, y, z = quat
    pitch = np.arcsin(np.clip(2 * (w * y - x * z), -1, 1))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    ang_vel_x = data.qvel[3]
    ang_vel_y = data.qvel[4]

    return body_z, pitch, roll, ang_vel_x, ang_vel_y


def main():
    model = mujoco.MjModel.from_xml_path("go1_envs/scenes/flat_scene.xml")
    data = mujoco.MjData(model)

    ctrl = PDController(n_joints=12, kp=50.0, kd=1.5)
    gait = BalanceGait(freq=3.0, target_height=0.27)
    keys = KeyState()

    print("Controls: W/S=fwd/back  A/D=turn  Q/E=lateral  P=walk/pause  SPACE=stop  R=reset")
    print("Robot starts standing still.")

    with mujoco.viewer.launch_passive(model, data, key_callback=keys.on_key) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        while viewer.is_running():
            step_start = time.time()
            vx, vy, yaw, walking, do_reset = keys.snapshot()

            if do_reset:
                mujoco.mj_resetDataKeyframe(model, data, 0)
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
            #电机pd ctrl
            ctrl.apply(ref, data)

            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

            # Status line
            if gait.is_recovering:
                print(f"\r[RECOVERING] pitch={np.degrees(pitch):+.1f}° roll={np.degrees(roll):+.1f}°", end="")
            elif walking:
                print(f"\r vx={vx:+.1f} vy={vy:+.1f}  pitch={np.degrees(pitch):+.1f}° roll={np.degrees(roll):+.1f}° z={body_z:.3f}m", end="")

            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
