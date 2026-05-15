import mujoco
import mujoco.viewer
import time
import numpy as np
import math

def main():
    # 1. 导入模型
    model_path = "mujoco_menagerie/unitree_go2/scene.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 2. 初始化机器狗姿态，防止出生时直接砸向地面
    # qpos 的前 7 个值是浮动基座的坐标和四元数：[x, y, z, qw, qx, qy, qz]
    data.qpos[:7] = np.array([0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0])
    
    # 基础站立角度 (髋、大腿、小腿)
    hip_base = 0.0
    thigh_base = 0.8
    calf_base = -1.5
    
    # 初始化各个关节的角度
    for leg in range(4):
        data.qpos[7 + leg*3] = hip_base
        data.qpos[7 + leg*3 + 1] = thigh_base
        data.qpos[7 + leg*3 + 2] = calf_base
        
    mujoco.mj_forward(model, data)

    # 3. 定义带有 PD 控制的步态生成器
    def update_control(t):
        freq = 6.0      # 迈步频率
        thigh_amp = 0.5 # 大腿摆动幅度
        calf_amp = 0.3  # 小腿抬起幅度

        # 步态相位差：右前(0)和左后(3)同步，左前(1)和右后(2)同步 (Trot步态)
        phases =[0.0, math.pi, math.pi, 0.0]
        
        # ⭐ 新增：PD 控制器参数 (根据Go2的重量和动力学调校)
        kp = 50.0  # 比例系数（刚度，类似于弹簧多硬）
        kd = 1.5   # 微分系数（阻尼，用于吸收震荡）

        for leg in range(4):
            phase = phases[leg]
            
            # 计算当前时间的运动学目标角度
            thigh_angle = thigh_base + thigh_amp * math.sin(t * freq + phase)
            calf_angle = calf_base + calf_amp * math.cos(t * freq + phase)
            
            target_pos =[hip_base, thigh_angle, calf_angle]
            
            # ⭐ 新增：遍历该腿的3个关节，执行 PD 力矩控制
            for joint_idx in range(3):
                # qpos 索引，7代表越过前面的浮动基座(pos+quat)
                q_idx = 7 + leg * 3 + joint_idx
                # qvel 索引，6代表越过前面的浮动基座速度(v+w)
                v_idx = 6 + leg * 3 + joint_idx
                
                # 获取该关节当前真实的【角度】和【角速度】
                curr_pos = data.qpos[q_idx]
                curr_vel = data.qvel[v_idx]
                
                # 计算需要的力矩: T = kp * (目标位置 - 当前位置) + kd * (目标速度 - 当前速度)
                # 这里我们期望的稳态目标速度为0，以简化公式
                torque = kp * (target_pos[joint_idx] - curr_pos) - kd * curr_vel
                
                # 将计算出的强大力矩下发给对应的电机
                data.ctrl[leg * 3 + joint_idx] = torque

    # 4. 启动可视化窗口并开启仿真主循环
    print("🚀 启动 MuJoCo 仿真，按 ESC 退出...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            t = data.time
            update_control(t)

            # 物理引擎步进计算
            mujoco.mj_step(model, data)

            # 让摄像机的视角动态跟随机器狗的三维位置
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

            # 保持仿真时间与现实时间一致
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()