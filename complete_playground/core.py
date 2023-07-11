"Core API for Environment"
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import numpy as np

from complete_playground import spaces
from complete_playground.utils import seeding

if TYPE_CHECKING:
    from complete_playground.envs.registration import EnvSpec

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Env(Generic[ObsType, ActType]):

    # Set this in SOME subclasses
    metadata: dict[str, Any] = {"render_modes": []}
    # define render_mode if your environment supports rendering
    render_mode: str | None = None
    reward_range = (-float("inf"), float("inf"))
    spec: EnvSpec | None = None

    # Set these in ALL subclasses
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]
    max_episode_steps: int | None = None

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
        """
        raise NotImplementedError
    
    def reset(
        self
    ) -> ObsType:
        """
        Resets the environment to an initial internal state, returning an initial observation.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment.

        Args:
            None
        """
        raise NotImplementedError
    

    def __str__(self):
        """Returns a string of the environment with :attr:`spec` id's if :attr:`spec.

        Returns:
            A string identifying the environment
        """
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"