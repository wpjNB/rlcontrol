"""Unified training entry point for Go1 environments.

Usage:
    python -m go1_envs.train balance [--timesteps N] [--n-envs N]
    python -m go1_envs.train walk [--timesteps N] [--n-envs N]
    python -m go1_envs.train jump [--timesteps N] [--n-envs N]
    python -m go1_envs.train eval <task> [--model PATH]
    python -m go1_envs.train viz <task> [--model PATH]
"""

import argparse

import gymnasium as gym

# Trigger env registration
#   train.py 导入 → import go1_envs → __init__.py 执行 → register() 注册三个环境 →
#   gym.make("Go1Balance-v0") 可用
import go1_envs  # noqa: F401

TASK_DEFAULTS = {
    "balance": {"timesteps": 300_000, "save_path": "go1_balance_ppo"},
    "walk": {"timesteps": 500_000, "save_path": "go1_walk_ppo"},
    "jump": {"timesteps": 800_000, "save_path": "go1_jump_ppo"},
}

TASK_ENV_IDS = {
    "balance": "Go1Balance-v0",
    "walk": "Go1Walk-v0",
    "jump": "Go1Jump-v0",
}


def make_env(task: str, render_mode=None):
    """Create a single env instance."""
    return gym.make(TASK_ENV_IDS[task], render_mode=render_mode)


def make_vec_env(task: str, n_envs: int = 1, render_mode=None):
    """Create vectorized env with normalization."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    if n_envs == 1:
        env = make_env(task, render_mode=render_mode)
    else:
        env = SubprocVecEnv([lambda: make_env(task) for _ in range(n_envs)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def train(task: str, timesteps: int, n_envs: int = 8):
    """Train a PPO agent on the specified task."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    save_path = TASK_DEFAULTS[task]["save_path"]

    train_env = make_vec_env(task, n_envs=n_envs)
    eval_env = make_vec_env(task, n_envs=1)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=42,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./{save_path}/",
        log_path=f"./{save_path}/logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Training {task} for {timesteps:,} steps with {n_envs} envs...")
    print(f"Save path: ./{save_path}/")
    print("-" * 60)

    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(f"{save_path}/final_model")
    train_env.save(f"{save_path}/vec_normalize.pkl")
    print(f"\nDone! Model saved to ./{save_path}/")

    train_env.close()
    eval_env.close()


def evaluate(task: str, model_path: str, num_episodes: int = 10):
    """Evaluate a trained model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecNormalize

    env = make_vec_env(task, n_envs=1)
    env = VecNormalize.load(f"{model_path}/../vec_normalize.pkl", env)
    env.training = False
    model = PPO.load(model_path, env=env)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=num_episodes, deterministic=True
    )
    print(f"Evaluation ({num_episodes} episodes): {mean_reward:.1f} +/- {std_reward:.1f}")
    env.close()


def visualize(task: str, model_path: str, num_episodes: int = 3):
    """Visualize a trained model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    env = make_vec_env(task, n_envs=1, render_mode="human")
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
        print(f"Episode {ep + 1}: Reward = {total_reward:.1f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Go1 RL Training")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("task", choices=TASK_DEFAULTS.keys())
    train_parser.add_argument("--timesteps", type=int, default=None)
    train_parser.add_argument("--n-envs", type=int, default=8)

    # Eval
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("task", choices=TASK_DEFAULTS.keys())
    eval_parser.add_argument("--model", type=str, default=None)
    eval_parser.add_argument("--episodes", type=int, default=10)

    # Visualize
    viz_parser = subparsers.add_parser("viz")
    viz_parser.add_argument("task", choices=TASK_DEFAULTS.keys())
    viz_parser.add_argument("--model", type=str, default=None)
    viz_parser.add_argument("--episodes", type=int, default=3)

    args = parser.parse_args()

    if args.command == "train":
        timesteps = args.timesteps or TASK_DEFAULTS[args.task]["timesteps"]
        train(args.task, timesteps, args.n_envs)
    elif args.command == "eval":
        save_path = TASK_DEFAULTS[args.task]["save_path"]
        model_path = args.model or f"./{save_path}/best_model"
        evaluate(args.task, model_path, args.episodes)
    elif args.command == "viz":
        save_path = TASK_DEFAULTS[args.task]["save_path"]
        model_path = args.model or f"./{save_path}/best_model"
        visualize(args.task, model_path, args.episodes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
