"""Visualize trained Go1WalkV2 model using MuJoCo viewer."""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import gymnasium as gym
import go1_envs  # noqa: F401 to register environments

MODEL_DIR = "go1_walkv2_ppo"

# Raw env for rendering
raw_env = gym.make("Go1Walk-v1", render_mode="human")
# VecEnv for model inference with normalization
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
vec_env = sb3_make_vec_env(lambda: gym.make("Go1Walk-v1"), n_envs=1)
vec_env = VecNormalize.load(f"{MODEL_DIR}/vec_normalize.pkl", vec_env)
vec_env.training = False

model = PPO.load(f"{MODEL_DIR}/best_model", env=vec_env)

num_episodes = 1000
try:
    for ep in range(num_episodes):
        raw_obs, _ = raw_env.reset()
        vec_env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            obs = vec_env.normalize_obs(raw_obs.reshape(1, -1))
            action, _ = model.predict(obs, deterministic=True)
            raw_obs, reward, terminated, truncated, _ = raw_env.step(action[0])
            vec_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        print(f"Episode {ep + 1}: 存活 {steps} 步, Reward = {total_reward:.1f}")
finally:
    raw_env.close()
    vec_env.close()
