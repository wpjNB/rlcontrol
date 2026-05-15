import gymnasium as gym
from stable_baselines3 import PPO

def main():
    env=gym.make("CartPole-v1")

    #训练模型
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=20000, progress_bar=True)
    # model.save("ppo_cartpole")
    
    # 加载预训练模
    model = PPO.load("ppo_cartpole")
    test_model(model)

def test_model(model):
    env=gym.make("CartPole-v1",render_mode="human")
    obs, _ = env.reset()
    done=False
    reward_sum = 0.0
    for _ in range(2000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward_sum += reward
        if done:
                print(reward_sum)
                reward_sum = 0.0
                obs, _ = env.reset()
    env.close()

if __name__ == "__main__":
    main()