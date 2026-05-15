"""Visualize Go1 environment with random actions using MuJoCo viewer."""

import time
import numpy as np
import mujoco
import mujoco.viewer

import gymnasium as gym
import go1_envs  # noqa: F401 to register environments

env = gym.make("Go1Walk-v1", render_mode="human")
obs, info = env.reset()

done=False
reward_sum = 0.0
try:
    for _ in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        time.sleep(0.01)
        done = terminated or truncated
        reward_sum += reward
        if done:
            print(reward_sum)
            reward_sum = 0.0
            obs, _ = env.reset()
finally:
    env.close()
