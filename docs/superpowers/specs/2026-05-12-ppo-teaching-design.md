# PPO 教学文档设计

## 目标
生成 `06_ppo_inverted_pendulum.py`，延续 `03_dqn_cartpole.py` 的教学风格。

## 环境
MuJoCo InvertedPendulum-v4（连续动作空间）

## 结构（6 模块）
1. 原理讲解（文件头注释）：REINFORCE → TRPO → PPO 演进
2. Actor-Critic 网络：Actor（策略）+ Critic（价值）
3. GAE 计算：Generalized Advantage Estimation
4. PPO 智能体：clipping、多 epoch 更新
5. 训练主程序：训练 + 评估
6. SB3 对比：Stable Baselines3 PPO 对比

## 设计约束
- 中文注释，公式旁标注代码位置
- 复用 REINFORCE 的 Policy_Network 结构
- 可直接运行
