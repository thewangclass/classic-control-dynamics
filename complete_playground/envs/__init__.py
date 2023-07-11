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