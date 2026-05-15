"""Go1 custom Gymnasium environments for reinforcement learning."""

from gymnasium.envs.registration import register

register(
    id="Go1Balance-v0",
    entry_point="go1_envs.go1_balance:Go1BalanceEnv",
    max_episode_steps=1000,
)

register(
    id="Go1Walk-v0",
    entry_point="go1_envs.go1_walk:Go1WalkEnv",
    max_episode_steps=1000,
)

register(
    id="Go1Jump-v0",
    entry_point="go1_envs.go1_jump:Go1JumpEnv",
    max_episode_steps=1000,
)

register(
    id="Go1Walk-v1",
    entry_point="go1_envs.go1_walk_v2:Go1WalkV2Env",
    max_episode_steps=1000,
)
