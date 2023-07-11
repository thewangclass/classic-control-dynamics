"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from complete_playground.envs.registration import(
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)


# Classic
# ----------------------------------------
register(
    id="CartPole-v0",
    entry_point="complete_playground.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="CartPole-v1",
    entry_point="complete_playground.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="MountainCar-v0",
    entry_point="complete_playground.envs.classic_control.mountain_car:MountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="Pendulum-v1",
    entry_point="complete_playground.envs.classic_control.pendulum:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="Acrobot-v1",
    entry_point="complete_playground.envs.classic_control.acrobot:AcrobotEnv",
    reward_threshold=-100.0,
    max_episode_steps=500,
)


# Hook to load plugins from entry points
load_plugin_envs()