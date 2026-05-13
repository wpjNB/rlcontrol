"""
PPO (Proximal Policy Optimization) on MuJoCo InvertedPendulum
从 REINFORCE 到 PPO 的完整演进：
理解为什么需要 PPO，以及它如何解决策略梯度的核心问题

================================================================
学习路线：你已经学过什么 → 这篇文档要学什么
================================================================

已学：
  02 Q-Learning     → 价值方法，离散动作，Q 表
  03 DQN            → 价值方法，离散动作，神经网络逼近 Q
  04 Blackjack       → Q-Learning 实战
  05 PPO (SB3)      → 调库使用，但不知道内部原理
  REINFORCE notebook → 策略梯度入门，但方差大、不稳定

本篇：
  06 PPO            → 策略梯度的"工业级"解决方案

================================================================
第一部分：为什么需要 PPO？（REINFORCE 的三大问题）
================================================================

REINFORCE 的核心公式：
    ∇J(θ) = E[∇log π(a|s) · G_t]

其中 G_t 是从时间步 t 开始的折扣累计回报（蒙特卡洛估计）。

问题 1：高方差
    G_t 用一次采样的实际回报来估计，每次采样轨迹不同，
    导致梯度估计方差极大，训练像"醉汉走路"。

    PPO 的解决：用 GAE（Generalized Advantage Estimation）
    结合 TD 误差和蒙特卡洛回报，降低方差。

问题 2：步长敏感
    策略更新太大 → 性能崩塌（新策略比旧策略差很多）
    策略更新太小 → 学习太慢
    REINFORCE 没有任何机制限制更新幅度。

    PPO 的解决：Clipping 机制，限制新旧策略的概率比在 [1-ε, 1+ε] 范围内。

问题 3：样本效率低
    REINFORCE 每条轨迹只用一次就丢弃。
    PPO 的解决：同一批数据重复训练多个 epoch。

================================================================
第二部分：从 TRPO 到 PPO 的演进
================================================================

TRPO（Trust Region Policy Optimization）的思路：
    在信任区域内最大化目标函数，用 KL 散度约束新旧策略的距离。
    数学上很优美，但实现复杂（需要计算 Fisher 信息矩阵、共轭梯度法）。

PPO 的思路（Schulman et al., 2017）：
    用一个简单的 clipping 操作代替 TRPO 的 KL 约束，
    效果接近 TRPO，但实现简单得多。

    核心公式（PPO-Clip）：
        r_t(θ) = π_θ(a|s) / π_θ_old(a|s)         ← 新旧策略的概率比
        L_CLIP = E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)]

    直觉理解：
        - 当 A_t > 0（这个动作比平均好）：鼓励多选这个动作，但 r_t 不超过 1+ε
        - 当 A_t < 0（这个动作比平均差）：鼓励少选这个动作，但 r_t 不低于 1-ε
        - ε 通常取 0.1~0.2，控制"信任区域"的大小

================================================================
第三部分：Actor-Critic 架构
================================================================

REINFORCE 只有 Actor（策略网络），用蒙特卡洛回报 G_t 作为梯度的权重。
PPO 使用 Actor-Critic：
    - Actor：策略网络 π(a|s)，输出动作分布的参数（均值、标准差）
    - Critic：价值网络 V(s)，估计状态价值函数

    优势函数 A(s,a) = Q(s,a) - V(s) 代替了 G_t
    其中 Q(s,a) 可以用 TD 估计：Q(s,a) ≈ r + γ·V(s')

    好处：Critic 提供了更好的 baseline，降低方差

================================================================
第四部分：GAE（Generalized Advantage Estimation）
================================================================

TD 误差：δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    低方差但高偏差（只看一步）

蒙特卡洛回报：G_t - V(s_t)
    低偏差但高方差（看完整轨迹）

GAE 的折中：
    A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}

    λ ∈ [0,1] 控制偏差-方差权衡：
        λ=0 → 纯 TD（低方差，高偏差）
        λ=1 → 纯蒙特卡洛（高方差，低偏差）
        λ=0.95 → 常用的折中值

================================================================
第五部分：PPO 完整算法流程
================================================================

for iteration = 1, 2, ... do:
    1. 用当前策略 π_θ 采集 N 条轨迹，存在缓冲区中
    2. 用 Critic V(s) 计算每个时间步的优势 A_t（GAE）
    3. 计算折扣回报目标：returns = advantages + V(s)
    4. 对缓冲区数据进行 K 个 epoch 的小批量更新：
        a. 计算概率比 r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
        b. 计算 clipped surrogate loss L_CLIP
        c. 计算 value loss = MSE(V(s), returns)
        d. 计算 entropy bonus（鼓励探索）
        e. 总损失 = -L_CLIP + c1·value_loss - c2·entropy
        f. 梯度下降更新 θ
    5. 更新旧策略：θ_old ← θ

================================================================
参考文献
================================================================
[1] Schulman et al., "Proximal Policy Optimization Algorithms", 2017
[2] Schulman et al., "High-Dimensional Continuous Control Using
    Generalized Advantage Estimation", ICLR 2016
[3] Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016
"""

import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


# ====================== 1. 超参数配置 ======================
@dataclass
class PPOConfig:
    """
    PPO 的所有超参数集中管理
    对比 REINFORCE：REINFORCE 只有 lr 和 gamma 两个参数
    PPO 需要更多参数来控制 clipping、GAE、多 epoch 更新等机制
    """
    # 环境
    env_id: str = "InvertedPendulum-v4"

    # 网络结构
    hidden_dim: int = 64             # 隐藏层维度（比 REINFORCE 的 16/32 大一些，任务更复杂）

    # PPO 核心参数
    lr: float = 3e-4                 # 学习率（比 REINFORCE 的 1e-4 稍大）
    gamma: float = 0.99              # 折扣因子（和 REINFORCE 相同）
    gae_lambda: float = 0.95         # GAE 的 λ 参数（控制偏差-方差权衡，REINFORCE 没有这个）
    clip_epsilon: float = 0.2        # PPO clipping 的 ε（核心参数，REINFORCE 没有这个）
    entropy_coef: float = 0.01       # 熵奖励系数（鼓励探索）
    value_coef: float = 0.5          # 价值损失系数
    max_grad_norm: float = 0.5       # 梯度裁剪阈值

    # 训练参数
    num_episodes: int = 1000         # 总训练 episode 数
    rollout_steps: int = 2048        # 每次采集的步数（PPO 按步采集，REINFORCE 按 episode）
    num_epochs: int = 10             # 每批数据重复训练的 epoch 数（PPO 的关键改进）
    batch_size: int = 64             # 小批量大小
    num_test_episodes: int = 10      # 测试 episode 数

    # 随机种子
    seed: int = 42


# ====================== 2. 网络定义 ======================
class ActorNetwork(nn.Module):
    """
    策略网络（Actor）：输出动作分布的参数

    对比 REINFORCE 的 Policy_Network：
        结构几乎相同（共享层 + 均值头 + 标准差头）
        区别在于 PPO 的 Actor 会被多次更新（多 epoch），而 REINFORCE 每 episode 只更新一次

    连续动作空间：输出正态分布的均值 μ 和标准差 σ
        a ~ N(μ(s), σ²)
        对数概率：log π(a|s) = -0.5 * [(a-μ)/σ]² - log(σ) - 0.5*log(2π)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 可学习的 log 标准差

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std)  # 保证标准差为正
        return mean, std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """返回正态分布对象，用于采样和计算概率"""
        mean, std = self.forward(obs)
        return Normal(mean, std)


class CriticNetwork(nn.Module):
    """
    价值网络（Critic）：估计状态价值 V(s)

    这是 REINFORCE 没有的！
    REINFORCE 用蒙特卡洛回报 G_t 作为梯度权重 → 高方差
    PPO 用 Critic 的 V(s) 计算优势函数 A(s,a) = Q(s,a) - V(s) → 低方差

    Critic 的输入是状态 s，输出是标量 V(s)（该状态的预期回报）
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # 输出标量 V(s)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ====================== 3. GAE 计算 ======================
def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    lam: float,
) -> list[float]:
    """
    计算 GAE（Generalized Advantage Estimation）

    公式：A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
    其中 TD 误差：δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)

    对比 REINFORCE：
        REINFORCE 用 G_t = Σ γ^k · r_{t+k}（完整轨迹回报）
        GAE 用 TD 误差的指数加权平均，λ 控制看多远

    代码逻辑（反向计算，从最后一个时间步开始）：
        advantage = δ_t + γ·λ·(1-done_t)·advantage_{t+1}
        这个递推关系等价于上面的求和公式，但更高效
    """
    advantages = []
    advantage = 0

    # 反向遍历，从最后一个时间步到第一个
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # 最后一步：没有下一个状态，用 0 作为 V(s_{T+1})
            next_value = 0
        else:
            next_value = values[t + 1]

        # TD 误差：δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
        # 公式对应：Schulman et al. 2016, Eq. 10
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # 递推计算 advantage：A_t = δ_t + γ·λ·(1-done)·A_{t+1}
        # 公式对应：Schulman et al. 2016, Eq. 11
        advantage = delta + gamma * lam * (1 - dones[t]) * advantage
        advantages.insert(0, advantage)

    return advantages


# ====================== 4. 经验缓冲区 ======================
@dataclass
class RolloutBuffer:
    """
    存储一次 rollout 采集的所有数据

    对比 REINFORCE：
        REINFORCE 只存 probs 和 rewards（两个列表）
        PPO 的缓冲区更复杂：需要存 obs、actions、log_probs、rewards、dones、values
        因为 PPO 要多次遍历这些数据（多 epoch 更新）
    """
    observations: list = None
    actions: list = None
    log_probs: list = None
    rewards: list = None
    dones: list = None
    values: list = None

    def __post_init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self):
        return len(self.rewards)


# ====================== 5. PPO 智能体 ======================
class PPOAgent:
    """
    PPO 智能体，包含 Actor 和 Critic

    对比 REINFORCE 类：
        REINFORCE：只有 Policy_Network（Actor），没有 Critic
        PPO：Actor + Critic，多了 clipping、GAE、多 epoch 更新

    关键区别总结：
    ┌─────────────┬──────────────────┬──────────────────────┐
    │             │ REINFORCE        │ PPO                  │
    ├─────────────┼──────────────────┼──────────────────────┤
    │ 网络        │ 只有 Actor       │ Actor + Critic       │
    │ 优势估计    │ 蒙特卡洛 G_t    │ GAE (TD 误差加权)    │
    │ 更新频率    │ 每 episode 一次  │ 每批数据 K 个 epoch  │
    │ 更新约束    │ 无               │ Clipping             │
    │ 样本效率    │ 低               │ 高（数据重复使用）   │
    │ 方差        │ 高               │ 低（Critic baseline） │
    └─────────────┴──────────────────┴──────────────────────┘
    """

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig):
        self.config = config
        self.device = torch.device("cpu")

        # Actor（策略网络）— 和 REINFORCE 的 Policy_Network 类似
        self.actor = ActorNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)

        # Critic（价值网络）— REINFORCE 没有这个
        self.critic = CriticNetwork(obs_dim, config.hidden_dim).to(self.device)

        # 优化器：分开优化 Actor 和 Critic（也可以共用一个）
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)

        # 经验缓冲区
        self.buffer = RolloutBuffer()

    def select_action(self, obs: np.ndarray):
        """
        根据当前策略选择动作

        和 REINFORCE 的 sample_action 几乎一样：
        1. 观测 → Actor → 均值和标准差
        2. 构建正态分布，采样动作
        3. 计算 log 概率（用于后续更新）

        区别：PPO 还会用 Critic 估计 V(s)，一起存入缓冲区
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Actor 输出动作分布
        dist = self.actor.get_distribution(obs_t)
        action = dist.sample()                    # 采样动作
        log_prob = dist.log_prob(action).sum()    # 对数概率（多维动作要 sum）

        # Critic 估计状态价值 V(s) — REINFORCE 没有这一步
        value = self.critic(obs_t)

        return (
            action.squeeze(0).detach().numpy(),
            log_prob.detach(),
            value.detach(),
        )

    def compute_returns_and_advantages(self):
        """
        计算 GAE 优势函数和折扣回报

        这是 PPO 相比 REINFORCE 最核心的改进之一：
        REINFORCE：直接用蒙特卡洛回报 G_t 作为梯度权重 → 高方差
        PPO：用 GAE 计算优势 A_t，再加回 V(s) 得到回报目标 → 低方差

        数学关系：
            A_t = GAE(γ,λ) 估计的是"这个动作比平均好多少"
            return_t = A_t + V(s_t) 作为 Critic 的训练目标
        """
        rewards = self.buffer.rewards
        values = self.buffer.values + [0]  # 最后补 0（终止状态的 V=0）
        dones = self.buffer.dones

        # 计算 GAE 优势
        advantages = compute_gae(rewards, values, dones,
                                 self.config.gamma, self.config.gae_lambda)

        # 计算回报目标 = advantages + V(s)
        # 这是 Critic 的训练目标：让 V(s) 逼近 return_t
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return advantages, returns

    def update(self):
        """
        PPO 的核心更新逻辑

        和 REINFORCE 的 update() 对比：
        ┌─────────────────────────────────────────────────────────────┐
        │ REINFORCE.update()                                          │
        │   1. 计算折扣回报 G_t                                       │
        │   2. loss = -Σ logπ(a|s) · G_t                              │
        │   3. 梯度下降（一次更新，用完整轨迹）                       │
        └─────────────────────────────────────────────────────────────┘
        ┌─────────────────────────────────────────────────────────────┐
        │ PPO.update()                                                │
        │   1. 计算 GAE 优势 A_t                                      │
        │   2. 将数据分成小批量                                       │
        │   3. 对每个小批量重复 K 个 epoch：                          │
        │      a. 计算新策略的概率比 r_t = π_new/π_old                │
        │      b. Clipped loss = min(r·A, clip(r)·A)                  │
        │      c. Value loss = MSE(V, returns)                        │
        │      d. Entropy bonus（鼓励探索）                           │
        │      e. 总损失 = -L_clip + c1·L_value - c2·entropy          │
        │   4. 更新网络参数                                           │
        └─────────────────────────────────────────────────────────────┘
        """
        cfg = self.config

        # 1. 计算优势和回报
        advantages, returns = self.compute_returns_and_advantages()

        # 转为 tensor
        obs_t = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions_t = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs_t = torch.stack(self.buffer.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # 优势标准化（降低方差，常用技巧）
        # REINFORCE 不需要这个，因为 G_t 本身已经有自然的尺度
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # 2. 多 epoch 更新（PPO 的关键改进，REINFORCE 没有）
        total_loss_actor = 0
        total_loss_critic = 0
        num_batches = 0

        for _ in range(cfg.num_epochs):
            # 随机打乱数据，生成小批量
            indices = np.arange(len(self.buffer))
            np.random.shuffle(indices)

            for start in range(0, len(self.buffer), cfg.batch_size):
                end = start + cfg.batch_size
                batch_idx = indices[start:end]

                batch_obs = obs_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                # ---- Actor 更新（PPO-Clip 核心） ----

                # 计算新策略的 log 概率
                # 公式：log π_θ(a|s)
                dist = self.actor.get_distribution(batch_obs)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                # 计算概率比 r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
                # 数学上：r_t = exp(log π_new - log π_old)
                # 公式对应：PPO 论文 Eq. 3
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                # Clipped surrogate loss（PPO 的灵魂）
                # L_CLIP = E[min(r·A, clip(r, 1-ε, 1+ε)·A)]
                # 公式对应：PPO 论文 Eq. 7
                #
                # 直觉：
                #   当 A > 0（好动作）：ratio 被限制在 [1, 1+ε]，鼓励多选但不过分
                #   当 A < 0（差动作）：ratio 被限制在 [1-ε, 1]，减少选择但不过分
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 熵奖励（鼓励探索）
                # 熵越大 → 分布越均匀 → 探索越多
                entropy = dist.entropy().mean()

                # ---- Critic 更新 ----
                # Critic 的目标：让 V(s) 逼近 return_t = A_t + V_old(s_t)
                value_pred = self.critic(batch_obs)
                critic_loss = nn.MSELoss()(value_pred, batch_returns)

                # ---- 总损失 ----
                # L = -L_CLIP + c1·L_value - c2·entropy
                # 公式对应：PPO 论文 Eq. 9
                loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy

                # 梯度更新
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪（防止梯度爆炸，PPO 的标准做法）
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_loss_actor += actor_loss.item()
                total_loss_critic += critic_loss.item()
                num_batches += 1

        # 清空缓冲区
        self.buffer.clear()

        return total_loss_actor / num_batches, total_loss_critic / num_batches


# ====================== 6. 训练主程序 ======================
def train_ppo(config: PPOConfig):
    """
    PPO 训练循环

    对比 REINFORCE 的训练循环：
    ┌──────────────────────────────────────────────────────────┐
    │ REINFORCE：每个 episode 结束后更新一次                   │
    │   for episode:                                           │
    │       收集完整轨迹 → update() → 清空缓冲区              │
    └──────────────────────────────────────────────────────────┘
    ┌──────────────────────────────────────────────────────────┐
    │ PPO：每收集 rollout_steps 步后更新一次                   │
    │   while total_steps < limit:                             │
    │       收集 2048 步 → compute GAE → update (10 epochs)    │
    │       注意：数据可以跨 episode 边界                      │
    └──────────────────────────────────────────────────────────┘
    """
    # 固定随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # 创建环境
    env = gym.make(config.env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"环境: {config.env_id}")
    print(f"状态维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    print(f"动作范围: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    print()

    # 创建 PPO 智能体
    agent = PPOAgent(obs_dim, action_dim, config)

    # 训练统计
    episode_rewards = []

    print("开始训练 PPO...")
    print("-" * 60)

    obs, _ = env.reset(seed=config.seed)
    total_steps = 0
    episode_reward = 0
    episode_count = 0

    while episode_count < config.num_episodes:
        # ---- 收集 rollout_steps 步的经验 ----
        for _ in range(config.rollout_steps):
            action, log_prob, value = agent.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存入缓冲区
            agent.buffer.add(obs, action, log_prob, reward, done, value.item())

            episode_reward += reward
            total_steps += 1
            obs = next_obs

            if done:
                episode_rewards.append(episode_reward)
                episode_count += 1

                # 打印进度
                if episode_count % 10 == 0:
                    avg = np.mean(episode_rewards[-10:])
                    print(f"Episode {episode_count:4d}/{config.num_episodes} | "
                          f"Steps: {total_steps:6d} | "
                          f"Reward: {episode_reward:7.1f} | "
                          f"Avg(10): {avg:7.1f}")

                episode_reward = 0
                obs, _ = env.reset()

        # ---- PPO 更新（REINFORCE 在这里只更新一次，PPO 更新 num_epochs 次） ----
        a_loss, c_loss = agent.update()
        if episode_count % 50 == 0 and episode_count > 0:
            print(f"  [Update] Actor Loss: {a_loss:.4f} | Critic Loss: {c_loss:.4f}")

    env.close()
    return agent, episode_rewards


# ====================== 7. 测试函数 ======================
def test_agent(agent: PPOAgent, config: PPOConfig):
    """测试训练好的智能体"""
    env = gym.make(config.env_id, render_mode=None)

    test_rewards = []
    for i in range(config.num_test_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            # 测试时用均值动作（不采样），更稳定
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.actor(obs_t)
            action = mean.squeeze(0).numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        test_rewards.append(episode_reward)
        print(f"  Test {i+1}: {episode_reward:.1f}")

    env.close()
    avg = np.mean(test_rewards)
    std = np.std(test_rewards)
    print(f"\n测试结果: {avg:.1f} ± {std:.1f}")
    print(f"（InvertedPendulum 满分 1000，通常 500+ 算学会）")
    return test_rewards


# ====================== 8. Stable Baselines3 对比 ======================
def compare_with_sb3():
    """
    用 Stable Baselines3 的 PPO 跑同一环境，对比代码量和性能

    这段代码展示：调库只需要几行，但理解原理后你能：
    1. 知道 SB3 背后做了什么
    2. 知道如何调参（哪些参数对应我们上面的哪些概念）
    3. 在 SB3 不支持的场景下自己实现
    """
    print("\n" + "=" * 60)
    print("Stable Baselines3 PPO 对比实验")
    print("=" * 60)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.evaluation import evaluate_policy
    except ImportError:
        print("未安装 stable-baselines3，跳过对比实验")
        print("安装命令: pip install stable-baselines3")
        return

    # SB3 的 PPO —— 只需要几行代码
    # 对比我们上面几百行的实现，这就是"调库"的力量
    # 但你现在知道这些参数背后是什么意思了！
    env = gym.make("InvertedPendulum-v4")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,          # 对应我们的 config.lr
        n_steps=2048,                # 对应我们的 config.rollout_steps
        batch_size=64,               # 对应我们的 config.batch_size
        n_epochs=10,                 # 对应我们的 config.num_epochs
        gamma=0.99,                  # 对应我们的 config.gamma
        gae_lambda=0.95,             # 对应我们的 config.gae_lambda
        clip_range=0.2,              # 对应我们的 config.clip_epsilon
        ent_coef=0.01,               # 对应我们的 config.entropy_coef
        vf_coef=0.5,                 # 对应我们的 config.value_coef
        max_grad_norm=0.5,           # 对应我们的 config.max_grad_norm
        verbose=0,
        seed=42,
    )

    print("SB3 PPO 训练中...")
    model.learn(total_timesteps=100_000)

    # 评估
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"SB3 PPO 测试结果: {mean_reward:.1f} ± {std_reward:.1f}")

    env.close()

    print("\n代码量对比:")
    print("  手写 PPO: ~350 行（理解每一行的数学原理）")
    print("  SB3 PPO:  ~15 行（调库，但你知道它在做什么）")
    print("\n参数对应关系:")
    print("  SB3 参数          →  我们的参数           →  论文概念")
    print("  learning_rate     →  config.lr            →  优化器学习率")
    print("  n_steps           →  config.rollout_steps  →  每次采集步数")
    print("  n_epochs          →  config.num_epochs     →  重复训练次数")
    print("  clip_range        →  config.clip_epsilon   →  PPO clipping ε")
    print("  gae_lambda        →  config.gae_lambda     →  GAE 的 λ")
    print("  ent_coef          →  config.entropy_coef   →  熵奖励系数")
    print("  vf_coef           →  config.value_coef     →  价值损失系数")


# ====================== 主程序入口 ======================
if __name__ == "__main__":
    config = PPOConfig()

    # 第一部分：手写 PPO 训练
    agent, rewards = train_ppo(config)

    # 第二部分：测试
    print("\n" + "=" * 60)
    print("测试手写 PPO 智能体")
    print("=" * 60)
    test_rewards = test_agent(agent, config)

    # 第三部分：SB3 对比
    compare_with_sb3()

    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)
    print("""
    你已经完成了从 REINFORCE 到 PPO 的完整学习路径：

    REINFORCE → PPO 的三大改进：
    1. GAE 优势估计（降低方差）
       - REINFORCE: G_t 蒙特卡洛回报
       - PPO: A_t = GAE(γ,λ) 折中 TD 和 MC

    2. Clipping 机制（限制更新幅度）
       - REINFORCE: 无约束，步长敏感
       - PPO: clip(r, 1-ε, 1+ε)，稳定更新

    3. 多 epoch 更新（提高样本效率）
       - REINFORCE: 每条轨迹只用一次
       - PPO: 同批数据重复训练 K 次

    下一步建议：
    - 尝试调整 gae_lambda（0.9→0.99）观察方差变化
    - 尝试调整 clip_epsilon（0.1→0.3）观察稳定性变化
    - 在更复杂的环境（如 HalfCheetah）上测试 PPO
    - 了解 PPO 的变体：PPO-Penalty、PPO-Lagrangian
    """)
