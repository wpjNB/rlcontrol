"""
加载自定义四足机器人环境并训练 PPO
===================================

本教程演示如何使用模型文件（`.xml` 格式）创建 mujoco 四足机器人行走环境，
并使用 PPO 算法训练智能体控制机器人行走。

步骤：

0. 准备机器人的 **MJCF**（或 **URDF**）模型文件。
    - 自行创建模型（参见 MuJoCo 教程），或
    - 使用现成模型（本教程使用 MuJoCo Menagerie 模型库中的 Unitree Go1）。
1. 使用 `xml_file` 参数加载模型。
2. 调整环境参数以获得期望行为。
    1. 调整仿真参数。
    2. 调整终止条件参数。
    3. 调整奖励参数。
    4. 调整观测参数。
3. 使用 PPO 训练智能体控制机器人移动。

使用方法：
    python load_quadruped_model.py test   - 验证环境
    python load_quadruped_model.py train  - 训练 PPO
    python load_quadruped_model.py eval   - 评估模型
    python load_quadruped_model.py viz    - 可视化
"""

# 读者应熟悉 `Gymnasium` API 和库、机器人学基础知识，
# 以及 `Gymnasium/MuJoCo` 环境及其所用机器人模型。
# 熟悉 **MJCF** 模型文件格式和 `MuJoCo` 模拟器并非必需，但建议了解。

# %%
# 环境准备
# --------
# 需要 `gymnasium>=1.0.0`。

import numpy as np

import gymnasium as gym

# 确保 Gymnasium 已正确安装，可在终端执行：
# pip install "gymnasium>=1.0.0"

# %%
# 步骤 0.1 - 下载机器人模型
# -------------------------
# 本教程从 MuJoCo Menagerie 模型库加载 Unitree Go1 机器人。
# Go1 是一款四足机器人，控制其行走是一项有挑战性的学习任务，
# 比 `Gymnasium/MuJoCo/Ant` 环境困难得多。
#
# 注：原始教程包含 Unitree Go1 机器人在平坦地形中的图片。
# 图片地址：https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_go1/go1.png?raw=true

# 可以下载整个 MuJoCo Menagerie 模型库（包含 `Go1`）：
# git clone https://github.com/google-deepmind/mujoco_menagerie.git

# 也可以使用其他四足机器人，只需为你的机器人调整环境参数即可。

# %%
# 步骤 1 - 加载模型
# -----------------
# 使用 `Ant-v5` 框架的 `xml_file` 参数加载模型。

# 基本加载方式（取消注释以使用）
# env = gym.make('Ant-v5', xml_file='./mujoco_menagerie/unitree_go1/scene.xml')

# 虽然这样就能加载模型，但还需要调整部分环境参数才能获得期望行为，
# 因此下面显式设置仿真、终止、奖励和观测参数，后续步骤中会逐一调整。

env = gym.make(
    "Ant-v5",
    xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
    forward_reward_weight=0,
    ctrl_cost_weight=0,
    contact_cost_weight=0,
    healthy_reward=0,
    main_body=1,
    healthy_z_range=(0, np.inf),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0,
    frame_skip=1,
    max_episode_steps=1000,
)

# %%
# 步骤 2 - 调整环境参数
# ---------------------
# 调整环境参数对获得期望的学习行为至关重要。
# 以下各小节中，建议读者查阅参数文档获取更详细信息。

# %%
# 步骤 2.1 - 调整仿真参数
# -----------------------
# 相关参数：`frame_skip`、`reset_noise_scale` 和 `max_episode_steps`。

# 调整 `frame_skip` 使 `dt` 达到合适值（典型值 `dt` ∈ [0.01, 0.1] 秒）。

# 提示：dt = frame_skip × model.opt.timestep，其中 `model.opt.timestep` 是
# MJCF 模型文件中设定的积分器时间步长。

# 我们使用的 `Go1` 模型积分器时间步长为 `0.002`，因此设置 `frame_skip=25`
# 可使 `dt` 为 `0.05s`。

# 为避免策略过拟合，`reset_noise_scale` 应设为与机器人尺寸相匹配的值，
# 尽可能大但不导致初始状态无效（无论控制动作如何都会触发终止），
# 对 `Go1` 选择 `0.1`。

# `max_episode_steps` 决定每回合在截断前的最大步数，
# 此处设为 1000 以与基础 `Gymnasium/MuJoCo` 环境保持一致，
# 如需更多步数可自行调大。

env = gym.make(
    "Ant-v5",
    xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
    forward_reward_weight=0,
    ctrl_cost_weight=0,
    contact_cost_weight=0,
    healthy_reward=0,
    main_body=1,
    healthy_z_range=(0, np.inf),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,  # 防止策略过拟合
    frame_skip=25,  # 设置 dt=0.05
    max_episode_steps=1000,  # 保持 1000 步
)

# %%
# 步骤 2.2 - 调整终止条件参数
# ---------------------------
# 终止条件对机器人环境很重要，可避免采样"无用"的时间步。

# 相关参数：`terminate_when_unhealthy` 和 `healthy_z_range`。

# 设置 `healthy_z_range` 使环境在机器人倒下或跳得过高时终止，
# 需要根据机器人身高选择合理的值，对 `Go1` 选择 `(0.195, 0.75)`。
# 注：`healthy_z_range` 检查的是机器人高度的绝对值，
# 如果场景包含不同海拔高度，应设为 `(-np.inf, np.inf)`。

# 也可设置 `terminate_when_unhealthy=False` 来完全禁用终止条件，
# 但这对 `Go1` 来说并不可取。

env = gym.make(
    "Ant-v5",
    xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
    forward_reward_weight=0,
    ctrl_cost_weight=0,
    contact_cost_weight=0,
    healthy_reward=0,
    main_body=1,
    healthy_z_range=(
        0.195,
        0.75,
    ),  # 避免机器人倒下或跳得过高时继续采样
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=1000,
)

# 注：如需自定义终止条件，可编写自己的 `TerminationWrapper`（参见文档）。

# %%
# 步骤 2.3 - 调整奖励参数
# -----------------------
# 相关参数：`forward_reward_weight`、`ctrl_cost_weight`、`contact_cost_weight`、
# `healthy_reward` 和 `main_body`。

# 对于 `forward_reward_weight`、`ctrl_cost_weight`、`contact_cost_weight` 和 `healthy_reward`，
# 需要为机器人选择合理的值，可参考默认 `MuJoCo/Ant` 的参数，
# 如有需要再进行调整。
# 对于 `Go1`，由于其执行器力矩范围更大，仅修改了 `ctrl_cost_weight`。

# 对于 `main_body`，需要指定哪个部件是主躯干
# （模型文件中通常命名为 "torso" 或 "trunk"）用于计算 `forward_reward`，
# 对于 `Go1` 是 `"trunk"`。
# （注：大多数情况下包括本例，可以保持默认值。）

env = gym.make(
    "Ant-v5",
    xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
    forward_reward_weight=1,  # 与 'Ant' 环境保持一致
    ctrl_cost_weight=0.05,  # 因 Go1 电机更强而调整
    contact_cost_weight=5e-4,  # 与 'Ant' 环境保持一致
    healthy_reward=1,  # 与 'Ant' 环境保持一致
    main_body=1,  # 对应 Go1 机器人的 "trunk"
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=1000,
)

# 注：如需自定义奖励函数，可编写自己的 `RewardWrapper`（参见文档）。

# %%
# 步骤 2.4 - 调整观测参数
# -----------------------
# 相关参数：`include_cfrc_ext_in_observation` 和
# `exclude_current_positions_from_observation`。

# 对于 `Go1` 没有特殊理由需要修改这些参数。

env = gym.make(
    "Ant-v5",
    xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,  # 与 'Ant' 环境保持一致
    exclude_current_positions_from_observation=False,  # 与 'Ant' 环境保持一致
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=1000,
)


# 注：如需额外的观测元素（如额外传感器），可编写自己的 `ObservationWrapper`（参见文档）。

# %%
# 步骤 3 - 训练智能体
# -------------------
# 到这里就完成了，可以使用强化学习算法训练智能体控制 `Go1` 机器人行走/奔跑。
# 注：如果你按照本教程使用了自己的机器人模型，可能会在训练过程中发现
# 某些环境参数不如预期，可随时返回步骤 2 进行调整。


def make_env(render_mode=None):
    """创建 Go1 四足机器人环境"""
    return gym.make(
        "Ant-v5",
        xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
        forward_reward_weight=1,
        ctrl_cost_weight=0.05,
        contact_cost_weight=5e-4,
        healthy_reward=1,
        main_body=1,
        healthy_z_range=(0.195, 0.75),
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=0.1,
        frame_skip=25,
        max_episode_steps=1000,
        render_mode=render_mode,
    )


def test_random_actions():
    """验证环境是否能正常加载和运行"""
    env = make_env(render_mode="human")
    obs, info = env.reset()
    total_reward = 0

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(f"环境测试成功！随机策略平均奖励: {total_reward:.1f}")


def make_vec_env(n_envs=1, render_mode=None):
    """创建向量化环境（并行运行多个环境实例，加速数据采集）"""
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    if n_envs == 1:
        env = make_env(render_mode=render_mode)
    else:
        # 多进程并行：每个环境在独立进程中运行
        # 8 个并行环境 ≈ 8 倍数据采集速度
        env = SubprocVecEnv([make_env for _ in range(n_envs)])

    # VecNormalize：自动归一化观测和奖励
    # 好处：训练更稳定、收敛更快
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def train_ppo(total_timesteps=500_000, save_path="go1_ppo", n_envs=8):
    """
    使用 PPO 训练 Go1 四足机器人

    加速方法：
    1. 并行环境（n_envs）：同时跑多个环境，数据采集速度 ×n_envs
    2. VecNormalize：归一化观测和奖励，收敛更快
    3. 减少 n_epochs：每批数据少训练几轮（10→5），更新更快
    4. GPU：如果有 CUDA GPU，SB3 会自动使用

    参数说明（对应 06_ppo_inverted_pendulum.py 中的概念）：
    - learning_rate=3e-4: 学习率
    - n_steps=2048: 每次采集的步数（rollout 长度）
    - batch_size=512: 小批量大小（并行环境数据更多，用更大的 batch）
    - n_epochs=5: 每批数据重复训练的次数（原 10，减半加速）
    - gamma=0.99: 折扣因子
    - gae_lambda=0.95: GAE 的 λ 参数
    - clip_range=0.2: PPO clipping 的 ε
    - ent_coef=0.001: 熵奖励系数
    - vf_coef=0.5: 价值损失系数
    - max_grad_norm=0.5: 梯度裁剪阈值
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    # 创建并行训练环境
    # n_envs=8 表示同时跑 8 个环境实例，数据采集速度提升约 8 倍
    train_env = make_vec_env(n_envs=n_envs)

    # 创建评估环境（用于定期评估和保存最佳模型）
    eval_env = make_vec_env(n_envs=1)

    # PPO 模型配置
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,              # 每个环境采集 2048 步，8 环境共 16384 步
        batch_size=512,            # 更大的 batch（数据量更大）
        n_epochs=5,                # 从 10 减到 5，更新速度翻倍
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=42,
    )

    # 评估回调：每 10000 步评估一次，保存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./{save_path}/",
        log_path=f"./{save_path}/logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"开始训练 Go1 四足机器人...")
    print(f"总训练步数: {total_timesteps:,}")
    print(f"模型保存路径: ./{save_path}/")
    print("-" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # 保存最终模型和 VecNormalize 统计量
    model.save(f"{save_path}/final_model")
    train_env.save(f"{save_path}/vec_normalize.pkl")
    print(f"\n训练完成！模型已保存到 ./{save_path}/")

    train_env.close()
    eval_env.close()

    return model


def evaluate_model(model_path="go1_ppo/best_model", num_episodes=10):
    """评估训练好的模型"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecNormalize

    env = make_vec_env(n_envs=1)
    # 加载训练时的归一化统计量
    env = VecNormalize.load(f"{model_path}/../vec_normalize.pkl", env)
    env.training = False  # 评估时不再更新统计量
    model = PPO.load(model_path, env=env)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=num_episodes, deterministic=True
    )

    print(f"评估结果 ({num_episodes} episodes):")
    print(f"  平均奖励: {mean_reward:.1f} ± {std_reward:.1f}")

    env.close()
    return mean_reward


def visualize_model(model_path="go1_ppo/best_model", num_episodes=3):
    """可视化训练好的模型（需要图形界面）"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    env = make_vec_env(n_envs=1, render_mode="human")
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
        print(f"Episode {ep+1}: Reward = {total_reward:.1f}")

    env.close()


def main():
    """
    Go1 四足机器人 PPO 训练主程序

    使用方法：
    1. 验证环境: python load_quadruped_model.py test
    2. 训练模型: python load_quadruped_model.py train
    3. 评估模型: python load_quadruped_model.py eval
    4. 可视化:   python load_quadruped_model.py viz
    """
    import sys

    # 从 mujoco_menagerie 加载 Go1 模型
    # 需要先克隆仓库: git clone https://github.com/google-deepmind/mujoco_menagerie.git
    if len(sys.argv) < 2:
        print("用法:")
        print("  python load_quadruped_model.py test   - 验证环境")
        print("  python load_quadruped_model.py train [步数] [并行数] - 训练 PPO")
        print("  python load_quadruped_model.py eval   - 评估模型")
        print("  python load_quadruped_model.py viz    - 可视化")
        print()
        print("示例:")
        print("  python load_quadruped_model.py train 500000 8   # 50万步，8并行环境")
        print("  python load_quadruped_model.py train 100000 4   # 10万步，4并行环境")
        return

    cmd = sys.argv[1]

    if cmd == "test":
        test_random_actions()
    elif cmd == "train":
        # 可通过命令行参数调整训练步数和并行环境数
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
        n_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        train_ppo(total_timesteps=steps, n_envs=n_envs)
    elif cmd == "eval":
        evaluate_model()
    elif cmd == "viz":
        visualize_model()
    else:
        print(f"未知命令: {cmd}")


if __name__ == "__main__":
    main()
