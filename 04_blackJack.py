from collections import defaultdict

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def run_episode(env, agent, train=True):
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        if train:
            agent.update(obs, action, reward, terminated, next_obs)
        episode_reward += reward
        done = terminated or truncated
        obs = next_obs
    return episode_reward


def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr),
        np.ones(window),
        mode=convolution_mode
    ) / window


def plot_training(env, agent, rolling_length=500):
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(agent.training_error, rolling_length, "valid")
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


def test_agent(agent, env, num_episodes=1000):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    total_rewards = [run_episode(env, agent, train=False) for _ in range(num_episodes)]

    agent.epsilon = old_epsilon

    total_rewards = np.array(total_rewards)
    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {np.mean(total_rewards > 0):.1%}")
    print(f"Average Reward: {np.mean(total_rewards):.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


if __name__ == "__main__":
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for episode in tqdm(range(n_episodes)):
        run_episode(env, agent, train=True)
        agent.decay_epsilon()

    plot_training(env, agent)
    test_agent(agent, env)
