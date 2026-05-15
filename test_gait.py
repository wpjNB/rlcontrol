"""Visualize TrotGait in MuJoCo viewer with PD control.

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

from go1_envs.gait import TrotGait, HOME_QPOS
from tools.pd_controller import PDController
from tools.key_state import KeyState


def main():
    model = mujoco.MjModel.from_xml_path("go1_envs/scenes/flat_scene.xml")
    data = mujoco.MjData(model)

    ctrl = PDController(n_joints=12, kp=50.0, kd=1.5)
    gait = TrotGait(freq=3.0)
    keys = KeyState()

    print(f"Gait: {gait}")
    print("Controls: W/S=fwd/back  A/D=turn  Q/E=lateral  P=walk/pause  SPACE=stop  R=reset")

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

            ref = gait.step(data.time, vx=vx, vy=vy, yaw_rate=yaw) if walking else HOME_QPOS
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
