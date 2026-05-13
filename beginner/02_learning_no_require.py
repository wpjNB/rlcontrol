import numpy as np
import random
from typing import Dict, List, Tuple

# ====================== 1. 环境模拟（网格世界） ======================
class GridWorldEnv:
    """简单的网格世界环境，用于演示Q-Learning"""
    def __init__(self, grid_size: Tuple[int, int] = (5, 5), 
                 start_pos: Tuple[int, int] = (0, 0), 
                 goal_pos: Tuple[int, int] = (4, 4),
                 obstacle_pos: List[Tuple[int, int]] = [(1, 1), (2, 2), (3, 1)]):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacle_pos = obstacle_pos
        self.current_pos = start_pos
        
        # 动作定义：0-上, 1-下, 2-左, 3-右
        self.actions = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.actions)
        
    def reset(self) -> int:
        """重置环境，返回初始状态的索引"""
        self.current_pos = self.start_pos
        return self.pos_to_state(self.current_pos)
    
    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        """将坐标位置转换为状态索引"""
        return pos[0] * self.grid_size[1] + pos[1]
    
    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """将状态索引转换为坐标位置"""
        return (state // self.grid_size[1], state % self.grid_size[1])
    
    def random_action(self) -> int:
        """随机选择一个动作（探索）"""
        return random.randint(0, self.num_actions - 1)
    
    def action_to_direction(self, action: int) -> str:
        """将动作索引转换为方向名称"""
        return self.actions[action]
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        执行动作，返回(next_state, reward, done)
        优化点：修复障碍物奖励无法触发的bug，简化逻辑判断
        """
        x, y = self.current_pos
        
        # 根据动作更新位置
        if action == 0:  # 上
            x = max(0, x - 1)
        elif action == 1:  # 下
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 2:  # 左
            y = max(0, y - 1)
        elif action == 3:  # 右
            y = min(self.grid_size[1] - 1, y + 1)
        
        # 检查是否碰到障碍物（核心修复：先记录是否碰到障碍物，再处理位置）
        new_pos = (x, y)
        hit_obstacle = False  # 标记是否碰到障碍物
        if new_pos in self.obstacle_pos:
            hit_obstacle = True  # 记录障碍物碰撞状态
            new_pos = self.current_pos  # 碰到障碍物，位置不变
        
        self.current_pos = new_pos
        next_state = self.pos_to_state(new_pos)
        
        # 计算奖励（基于提前记录的hit_obstacle，修复原逻辑矛盾）
        if new_pos == self.goal_pos:
            reward = 100.0  # 到达终点，大奖励
            done = True
        elif hit_obstacle:  # 基于标记判断，而非修改后的new_pos
            reward = -50.0  # 碰到障碍物，惩罚
            done = False
        else:
            reward = -1.0  # 每走一步小惩罚，鼓励尽快到达终点
            done = False
        
        return next_state, reward, done

# ====================== 2. Q-Learning 主程序 ======================
if __name__ == "__main__":
    # 优化点1：固定随机种子，保证实验结果可复现
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 初始化环境
    env = GridWorldEnv(
        grid_size=(5, 5),          # 5x5网格
        start_pos=(0, 0),          # 起点
        goal_pos=(4, 4),           # 终点
        obstacle_pos=[(1,1), (2,2), (3,1)]  # 障碍物位置
    )
    
    # 计算状态数量
    num_states = env.grid_size[0] * env.grid_size[1]
    num_actions = env.num_actions
    
    # 初始化Q表（动作价值函数），形状为 [状态数量，动作数量]
    Q_table = np.zeros([num_states, num_actions])
    
    # train过程中的定义超参数
    learning_rate = 0.1       # 学习率
    discount_factor = 0.9     # 折扣因子
    epsilon = 0.1             # 探索率
    total_episodes = 1000     # 训练轮数
    
    # 训练过程
    for episode in range(total_episodes):
        state = env.reset()    # 重置环境到起点，获取初始状态 S
        done = False           # 标记本轮是否结束
        total_reward = 0       # 记录本轮总奖励
        
        while not done:
            # 1. ε-贪婪策略选择动作
            if random.uniform(0, 1) < epsilon:
                action = env.random_action()  # 探索：随机选动作
            else:
                # 利用：选Q值最高的动作，处理平局情况
                q_values = Q_table[state]
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                action = random.choice(best_actions)  # 平局时随机选一个
            
            # 2. 执行动作，与环境交互
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 3. 更新Q表（核心：Q-Learning公式）
            # Q(S, A) = Q(S, A) + α * [ R + γ * max(Q(S', a')) - Q(S, A) ]
            old_value = Q_table[state, action]
            next_max = np.max(Q_table[next_state])  # 下一状态的最大Q值
            
            # 计算目标Q值
            target = reward + discount_factor * next_max
            # 更新Q值
            new_value = old_value + learning_rate * (target - old_value)
            Q_table[state, action] = new_value
            
            # 4. 进入下一个状态
            state = next_state
        
        # 每100轮打印一次训练进度
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{total_episodes}, Total Reward: {total_reward:.1f}")
    
    # ====================== 3. 提取最优策略（优化打印格式，更直观） ======================
    policy: Dict[int, str] = {}
    print("\n=== 学习完成后的最优策略（按网格排版）===")
    
    # 优化点2：按网格形状打印，直观展示整个网格的策略
    grid_rows, grid_cols = env.grid_size
    for row in range(grid_rows):
        row_str = []
        for col in range(grid_cols):
            pos = (row, col)
            state = env.pos_to_state(pos)
            if pos in env.obstacle_pos:
                row_str.append("  障碍物  ")
            elif pos == env.goal_pos:
                row_str.append("   终点   ")
            elif pos == env.start_pos:
                # 同时标记起点和其最优动作
                best_action_idx = np.argmax(Q_table[state])
                best_action = env.action_to_direction(best_action_idx)
                row_str.append(f" 起点({best_action}) ")
            else:
                best_action_idx = np.argmax(Q_table[state])
                best_action = env.action_to_direction(best_action_idx)
                policy[state] = best_action
                row_str.append(f"   {best_action}   ")
        print("|".join(row_str))
    
    # ====================== 4. 测试最优策略 ======================
    print("\n=== 测试最优策略 ===")
    state = env.reset()
    done = False
    steps = 0
    path = [env.state_to_pos(state)]
    
    while not done and steps < 50:  # 最多50步，防止无限循环
        best_action_idx = np.argmax(Q_table[state])  #选择最好的动作
        next_state, reward, done = env.step(best_action_idx) #执行动作，返回观测值（下一步动作）
        current_pos = env.state_to_pos(next_state)
        path.append(current_pos)
        state = next_state
        steps += 1
    
    print(f"路径: {path}")
    print(f"到达终点步数: {steps}")
    print(f"是否到达终点: {env.current_pos == env.goal_pos}")

#智能体（agent），动作（action），环境（env），奖励（reward），状态（state）
#策略（policy）
#价值函数（value function）