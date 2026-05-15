"""Unified training entry point for Go1 environments.

Usage:
    python -m go1_envs.train train <task> [--timesteps N] [--n-envs N]
    python -m go1_envs.train eval <task> [--model PATH]
    python -m go1_envs.train viz <task> [--model PATH]
    python -m go1_envs.train record <task> [--model PATH] [--output FILE]
"""

import argparse
import os

import gymnasium as gym

# Trigger env registration
#   train.py 导入 → import go1_envs → __init__.py 执行 → register() 注册三个环境 →
#   gym.make("Go1Balance-v0") 可用
import go1_envs  # noqa: F401

TASK_DEFAULTS = {
    "balance": {"timesteps": 300_000, "save_path": "go1_balance_ppo"},
    "walk": {"timesteps": 500_000, "save_path": "go1_walk_ppo"},
    "walkv2": {"timesteps": 200_000, "save_path": "go1_walkv2_ppo"},
    "jump": {"timesteps": 800_000, "save_path": "go1_jump_ppo"},
}

# Curriculum schedule: list of (difficulty, timesteps) pairs
WALKV2_CURRICULUM = [
    (0.3, 500_000),   # Phase 1: small forward commands, light penalties
    (0.6, 1_000_000),   # Phase 2: medium commands
    (1.0, 1_500_000), # Phase 3: full task
]

TASK_ENV_IDS = {
    "balance": "Go1Balance-v0",
    "walk": "Go1Walk-v0",
    "walkv2": "Go1Walk-v1",
    "jump": "Go1Jump-v0",
}


class _DifficultyForwarder(gym.Wrapper):
    """Forward set_difficulty to the underlying env."""
    def set_difficulty(self, d):
        self.env.unwrapped.set_difficulty(d)

def make_env(task: str, render_mode=None, difficulty=1.0):
    """Create a single env instance."""
    import go1_envs  # noqa: F401 ensure registration in subprocesses
    env = gym.make(TASK_ENV_IDS[task], render_mode=render_mode, difficulty=difficulty)
    return _DifficultyForwarder(env)


def make_vec_env(task: str, n_envs: int = 1, render_mode=None, difficulty=1.0):
    """Create vectorized env with normalization."""
    from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

    env = sb3_make_vec_env(
        lambda: make_env(task, render_mode=render_mode, difficulty=difficulty),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def _set_vec_env_difficulty(vec_env, difficulty: float):
    """Update difficulty on all sub-envs of a VecEnv."""
    vec_env.env_method("set_difficulty", difficulty)


def train_curriculum(task: str, curriculum=None, n_envs: int = 8, resume: bool = False):
    """Train walkv2 with curriculum: gradually increase difficulty.

    If resume=True, load the existing model and continue from the last phase.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import VecNormalize

    if curriculum is None:
        curriculum = WALKV2_CURRICULUM

    save_path = TASK_DEFAULTS[task]["save_path"]
    total_ts = sum(ts for _, ts in curriculum)

    # Start with first difficulty level
    d0, _ = curriculum[0]
    train_env = make_vec_env(task, n_envs=n_envs, difficulty=d0)
    eval_env = make_vec_env(task, n_envs=1, render_mode="human", difficulty=d0)

    if resume:
        model_file = f"{save_path}/final_model.zip"
        vec_file = f"{save_path}/vec_normalize.pkl"
        if not os.path.exists(model_file):
            print(f"Error: model not found at {model_file}")
            return
        if not os.path.exists(vec_file):
            print(f"Error: vec_normalize not found at {vec_file}")
            return

        # Reload envs with saved normalization
        train_env.close()
        eval_env.close()
        train_env = VecNormalize.load(vec_file, make_vec_env(task, n_envs=n_envs, difficulty=d0))
        eval_env = VecNormalize.load(vec_file, make_vec_env(task, n_envs=1, render_mode="human", difficulty=d0))
        train_env.training = True
        eval_env.training = False

        model = PPO.load(model_file, env=train_env)
        prev_ts = model.num_timesteps
        print(f"Resumed from {model_file} (previous timesteps: {prev_ts:,})")
    else:
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
        prev_ts = 0

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./{save_path}/",
        log_path=f"./{save_path}/logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Curriculum training for {task}:")
    for i, (d, ts) in enumerate(curriculum):
        print(f"  Phase {i+1}: difficulty={d:.1f}, timesteps={ts:,}")
    print(f"  Total: {total_ts:,} steps (+ previous {prev_ts:,})")
    print(f"  Save path: ./{save_path}/")
    print("-" * 60)

    trained_ts = 0
    for i, (d, ts) in enumerate(curriculum):
        print(f"\n=== Phase {i+1}/{len(curriculum)}: difficulty={d:.1f} ({ts:,} steps) ===")
        _set_vec_env_difficulty(train_env, d)
        _set_vec_env_difficulty(eval_env, d)

        model.learn(
            total_timesteps=prev_ts + trained_ts + ts,
            callback=eval_callback,
            progress_bar=True,
            reset_num_timesteps=False if resume else (i == 0),
        )
        trained_ts += ts

    model.save(f"{save_path}/final_model")
    train_env.save(f"{save_path}/vec_normalize.pkl")
    print(f"\nDone! Model saved to ./{save_path}/")

    train_env.close()
    eval_env.close()


def train(task: str, timesteps: int, n_envs: int = 8, resume: bool = False):
    """Train a PPO agent on the specified task.

    If resume=True, load the existing model and vec_normalize from save_path
    and continue training for additional timesteps.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import VecNormalize

    save_path = TASK_DEFAULTS[task]["save_path"]

    if resume:
        model_file = f"{save_path}/final_model.zip"
        vec_file = f"{save_path}/vec_normalize.pkl"
        if not os.path.exists(model_file):
            print(f"Error: model not found at {model_file}")
            return
        if not os.path.exists(vec_file):
            print(f"Error: vec_normalize not found at {vec_file}")
            return

        train_env = VecNormalize.load(vec_file, make_vec_env(task, n_envs=n_envs))
        eval_env = VecNormalize.load(vec_file, make_vec_env(task, n_envs=1, render_mode="human"))
        train_env.training = True
        eval_env.training = False

        model = PPO.load(model_file, env=train_env)
        print(f"Resumed from {model_file}")
        print(f"Previous timesteps: {model.num_timesteps:,}")
    else:
        train_env = make_vec_env(task, n_envs=n_envs)
        eval_env = make_vec_env(task, n_envs=1, render_mode="human")

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
        total_timesteps=model.num_timesteps + timesteps,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=not resume,
    )

    model.save(f"{save_path}/final_model")
    train_env.save(f"{save_path}/vec_normalize.pkl")
    print(f"\nDone! Model saved to ./{save_path}/")

    train_env.close()
    eval_env.close()


def evaluate(task: str, model_path: str, num_episodes: int = 10):
    """Evaluate a trained model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    import os
    env = make_vec_env(task, n_envs=1)
    vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    env = VecNormalize.load(vec_path, env)
    env.training = False
    model = PPO.load(model_path, env=env)

    # Manual evaluation to inspect reward terms
    all_rewards = []
    all_lengths = []
    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0
        ep_len = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_len += 1
            if ep_len >= 1000:
                break
        all_rewards.append(ep_reward)
        all_lengths.append(ep_len)
        print(f"  Episode {ep+1}: reward={ep_reward:.2f}, length={ep_len}")

    import numpy as np
    print(f"\nEvaluation ({num_episodes} episodes): "
          f"{np.mean(all_rewards):.1f} +/- {np.std(all_rewards):.1f} | "
          f"avg length: {np.mean(all_lengths):.0f}")
    env.close()


def visualize(task: str, model_path: str, num_episodes: int = 3):
    """Visualize a trained model (requires display)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    import time
    # Raw env for rendering
    raw_env = gym.make(TASK_ENV_IDS[task], render_mode="human")
    # VecEnv for model inference with normalization
    vec_env = make_vec_env(task, n_envs=1)
    vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    vec_env = VecNormalize.load(vec_path, vec_env)
    vec_env.training = False
    model = PPO.load(model_path, env=vec_env)

    for ep in range(num_episodes):
        raw_obs, _ = raw_env.reset()
        vec_env.reset()
        total_reward = 0
        done = False
        while not done:
            obs = vec_env.normalize_obs(raw_obs.reshape(1, -1))
            action, _ = model.predict(obs, deterministic=True)
            raw_obs, reward, terminated, truncated, _ = raw_env.step(action[0])
            vec_env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {ep + 1}: Reward = {total_reward:.1f}")
    raw_env.close()
    vec_env.close()


def record(task: str, model_path: str, num_episodes: int = 1, output: str = "eval.mp4"):
    """Record evaluation video (no display needed)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    import imageio

    env = make_vec_env(task, n_envs=1, render_mode="rgb_array")
    vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    env = VecNormalize.load(vec_path, env)
    env.training = False
    model = PPO.load(model_path, env=env)

    frames = []
    for ep in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        print(f"Episode {ep + 1}: Reward = {total_reward:.1f}")

    if frames:
        imageio.mimsave(output, frames, fps=20)
        print(f"\nVideo saved to {output} ({len(frames)} frames)")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Go1 RL Training")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("task", choices=TASK_DEFAULTS.keys())
    train_parser.add_argument("--timesteps", type=int, default=None)
    train_parser.add_argument("--n-envs", type=int, default=8)
    train_parser.add_argument("--resume", action="store_true", help="Resume training from saved model")

    # Curriculum train (walkv2 only)
    curr_parser = subparsers.add_parser("curriculum")
    curr_parser.add_argument("--n-envs", type=int, default=8)
    curr_parser.add_argument("--resume", action="store_true", help="Resume curriculum from saved model")

    # Eval
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("task", choices=TASK_DEFAULTS.keys())
    eval_parser.add_argument("--model", type=str, default=None)
    eval_parser.add_argument("--episodes", type=int, default=10)

    # Visualize
    viz_parser = subparsers.add_parser("viz")
    viz_parser.add_argument("task", choices=TASK_DEFAULTS.keys())
    viz_parser.add_argument("--model", type=str, default=None)
    viz_parser.add_argument("--episodes", type=int, default=1000)

    # Record video
    rec_parser = subparsers.add_parser("record")
    rec_parser.add_argument("task", choices=TASK_DEFAULTS.keys())
    rec_parser.add_argument("--model", type=str, default=None)
    rec_parser.add_argument("--episodes", type=int, default=1)
    rec_parser.add_argument("--output", type=str, default="eval.mp4")

    args = parser.parse_args()

    if args.command == "train":
        timesteps = args.timesteps or TASK_DEFAULTS[args.task]["timesteps"]
        train(args.task, timesteps, args.n_envs, resume=args.resume)
    elif args.command == "curriculum":
        train_curriculum("walkv2", n_envs=args.n_envs, resume=args.resume)
    elif args.command == "eval":
        save_path = TASK_DEFAULTS[args.task]["save_path"]
        model_path = args.model or f"./{save_path}/best_model"
        if not os.path.dirname(model_path):
            model_path = f"./{save_path}/{model_path}"
        evaluate(args.task, model_path, args.episodes)
    elif args.command == "viz":
        save_path = TASK_DEFAULTS[args.task]["save_path"]
        model_path = args.model or f"./{save_path}/best_model"
        if not os.path.dirname(model_path):
            model_path = f"./{save_path}/{model_path}"
        visualize(args.task, model_path, args.episodes)
    elif args.command == "record":
        save_path = TASK_DEFAULTS[args.task]["save_path"]
        model_path = args.model or f"./{save_path}/best_model"
        if not os.path.dirname(model_path):
            model_path = f"./{save_path}/{model_path}"
        record(args.task, model_path, args.episodes, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
