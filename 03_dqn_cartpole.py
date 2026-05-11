"""
DQN (Deep Q-Network) on CartPole
从 Q-Learning 到深度强化学习的第一步：
用神经网络替代 Q 表，处理连续状态空间

对比 02_learning.py 的变化：
- Q 表 → 神经网络（状态 → 各动作的 Q 值）
- 离散格子坐标 → 连续状态向量（位置、速度、角度、角速度）
- 新增：经验回放、目标网络
"""

import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ====================== 1. Q 网络（替代 Q 表） ======================
class QNetwork(nn.Module):
    """
    用神经网络逼近 Q 函数
    输入：状态向量（CartPole 是 4 维：位置、速度、角度、角速度）
    输出：每个动作的 Q 值（CartPole 是 2 个动作：左推、右推）

    对比 02_learning.py：
    - Q 表：Q_table[state, action] = 值  （查表）
    - DQN：  Q_network(state) → [Q左, Q右] （神经网络输出）
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ====================== 2. 经验回放缓冲区（DQN 新增） ======================
class ReplayBuffer:
    """
    存储历史经验 (s, a, r, s', done)，训练时随机采样
    作用：打破数据相关性，稳定训练

    02_learning.py 没有这个——Q-Learning 每步直接更新，不需要存储历史
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ====================== 3. DQN 智能体 ======================
class DQNAgent:
    """
    核心思想和 Q-Learning 一样，但用神经网络代替 Q 表

    两个关键改进：
    1. 经验回放：从历史中随机抽样训练，打破时间相关性
    2. 目标网络：用一个延迟更新的网络计算目标 Q 值，稳定训练
    """
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim
        self.device = torch.device("cpu")

        # 两个网络：policy_net（主网络，实时更新）和 target_net（目标网络，延迟更新）
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 初始权重相同
        self.target_net.eval()  # 目标网络不参与训练，只用于计算目标 Q 值

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer(capacity=10000)

        # 超参数（对应 02_learning.py 的超参数）
        self.batch_size = 64        # 每次从回放缓冲区抽 64 条经验训练
        self.gamma = 0.99           # 折扣因子（02_learning.py 中是 0.9）
        self.epsilon = 1.0          # 探索率初始值（从高到低逐步减少）
        self.epsilon_min = 0.01     # 探索率下限
        self.epsilon_decay = 0.998  # 探索率衰减（更慢衰减，多探索）
        self.target_update_freq = 5  # 每 5 个 episode 更新一次目标网络

    def select_action(self, state: np.ndarray) -> int:
        """
        ε-贪婪策略（和 02_learning.py 逻辑一样）
        区别：利用时用神经网络输出 Q 值，而非查 Q 表
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 探索

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()  # 利用

    def train_step(self):
        """
        从回放缓冲区采样，更新 Q 网络
        对应 02_learning.py 的 Q 表更新公式：
            Q(S,A) = Q(S,A) + α * [R + γ * max(Q(S',a')) - Q(S,A)]

        DQN 版本：
            loss = (target - Q(s,a))²
            target = R + γ * max(Q_target(s',a'))   (用目标网络计算)
        """
        if len(self.buffer) < self.batch_size * 5:  # 预热：先攒够足够数据
            return

        # 1. 从回放缓冲区随机采样
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 2. 计算当前 Q 值：Q(s, a)
        q_values = self.policy_net(states_t)
        q_current = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # 3. 计算目标 Q 值：R + γ * max(Q_target(s', a'))
        #    注意：用 target_net 而非 policy_net，这是 DQN 的关键
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + self.gamma * q_next * (1 - dones_t)

        # 4. 更新网络（梯度下降）
        loss = nn.MSELoss()(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)  # 梯度裁剪，防止爆炸
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """将 policy_net 的权重复制到 target_net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """逐步降低探索率（从 1.0 降到 0.01）"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ====================== 4. 训练主程序 ======================
if __name__ == "__main__":
    # 固定随机种子
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 创建 CartPole 环境
    # CartPole：一根杆子放在小车上，通过左右推保持平衡
    # 状态：[车位置, 车速度, 杆角度, 杆角速度]（4 维连续值）
    # 动作：0=向左推, 1=向右推（2 个离散动作）
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n              # 2

    print(f"状态维度: {state_dim}")
    print(f"动作数量: {action_dim}")
    print(f"状态示例: {env.reset()[0]}")
    print()

    # 创建智能体
    agent = DQNAgent(state_dim, action_dim)

    # 训练参数
    total_episodes = 1000
    scores = []  # 记录每轮总奖励

    print("开始训练...")
    print("-" * 50)

    for episode in range(total_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 1. 选动作（ε-贪婪）
            action = agent.select_action(state)

            # 2. 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # 3. 存入经验回放缓冲区
            agent.buffer.push(state, action, reward, next_state, done)

            # 4. 训练网络
            agent.train_step()

            state = next_state

        # 更新探索率
        agent.decay_epsilon()

        # 定期更新目标网络
        if (episode + 1) % agent.target_update_freq == 0:
            agent.update_target()

        scores.append(total_reward)

        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode+1:4d}/{total_episodes} | "
                  f"Score: {total_reward:.0f} | "
                  f"Avg(50): {avg_score:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    env.close()

    # ====================== 5. 测试训练好的智能体 ======================
    print("\n" + "=" * 50)
    print("测试训练好的智能体...")
    print("=" * 50)

    env = gym.make("CartPole-v1", render_mode=None)
    agent.epsilon = 0.0  # 关闭探索，纯利用

    test_scores = []
    for i in range(10):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        test_scores.append(total_reward)
        print(f"  Test {i+1}: {total_reward:.0f}")

    env.close()
    print(f"\n平均得分: {np.mean(test_scores):.1f} / 500")
    print(f"（满分 500，通常 200+ 算学会，400+ 算很好）")
