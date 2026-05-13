import gymnasium as gym
import numpy as np

# 图形化模式，直接弹出 MuJoCo 窗口
env = gym.make("Ant-v5", render_mode="human")
obs, info = env.reset()

print(f"观测空间: {env.observation_space.shape}")
print(f"动作空间: {env.action_space.shape}")

total_reward = 0

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step % 100 == 0:
        print(f"Step {step:3d} | Reward: {reward:.4f} | Total: {total_reward:.4f}")

    if terminated or truncated:
        print(f"Episode ended at step {step}")
        obs, info = env.reset()
        total_reward = 0

env.close()
print("Done!")
