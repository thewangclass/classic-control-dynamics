__credits__ = ["Carlos Luis"]
import sys
sys.path.append("/home/thewangclass/projects/classic-control-dynamics/")
from os import path
from typing import Optional

import numpy as np

class Pendulum():
    """
    ## Description

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](/_static/diagrams/pendulum.png)

    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |


    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `$\theta$` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ## Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.

    ## Arguments

    - `g`: acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
      The default value is g = 10.0 .

    ```python
    import gymnasium as gym
    gym.make('Pendulum-v1', g=9.81)
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.

    ## Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self):
        ##################################################
        # EPISODE ENDING
        ##################################################
        self.max_episode_steps = 200    # going over this causes truncation
        self.steps = 0
        self.steps_beyond_terminated = None


        ##################################################
        # SYSTEM DIMENSIONS
        ##################################################
        self.max_speed = 8
        self.max_torque = 2.0
        self.tau = 0.05
        self.gravity = 10.0
        self.mass = 1.0
        self.length = 1.0

        ##################################################
        # CONSTRAINTS
        ##################################################
        self.upper_bound = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the gymnasium api
        self.lower_bound = -self.upper_bound


        ##################################################
        # DEFINE ACTION AND OBSERVATION SPACE
        ##################################################
        # Possible actions the cartpole can take
        # 0: push cart to left
        # 1: push cart to right
        self.action_space = np.array([-self.max_torque, self.max_torque], dtype=np.float32)
        self.action_type = "Box"  # used in buffer to determine shape of memory
        self.observation_space = self.upper_bound # to use in network for first layer input
        self.state = None

        ##################################################
        # INITIALIZE EPISODIC REWARD AND LENGTH
        ##################################################
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        ##################################################
        # INITIALIZE NEW RANDOM STARTING STATE
        ##################################################
        high = np.array([np.pi, 1.0], dtype=np.float32)
        low = -high  
        self.state = np.random.uniform(low=low, high=high, size=(2,)).astype(np.float32)


        ##################################################
        # RESET EPISODIC VALUES
        ##################################################   
        self.steps = 0
        self.episode_reward = 0
        self.episode_length = 0

        return np.array([self._get_obs()])
    

    def step(self, action):
        theta, theta_dot = self.state 

        # update state
        g = self.gravity
        m = self.mass
        l = self.length
        dt = self.tau
        torque = np.clip(action, -self.max_torque, self.max_torque)[0]
        costs = angle_normalize(theta) ** 2 + 0.1 * theta_dot**2 + 0.001 * (torque**2)
        newthdot = theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * torque) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = theta + newthdot * dt
        self.state = np.array([newth, newthdot])

        # there is no termination possibility
        terminated = False
        reward = -costs

        # check truncation
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps

        # update episode reward and length
        self.episode_reward += reward
        self.episode_length += 1

        # create info to return
        info = {}
        if terminated or truncated:
            # See gymnasium wrappers record_episode_statistics.py
            info['final_observation'] = self.state
            info['_final_observation'] = np.array(True, dtype=bool)
            info['final_info'] = {
                'episode': {
                    'r': np.array(
                        np.array([self.episode_reward]),
                        dtype=np.float32
                    ),
                    'l': np.array(
                        np.array([self.episode_length]),
                        dtype=np.int32
                    ),
                    't': 'unassigned for now: elapsed time since beginning of episode'
                }
            }
            info['_final_info'] = np.array(True, dtype=bool)
            
            if truncated:
                info['TimeLimit.truncated'] = True

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi



if __name__ == '__main__':
    env = Pendulum()
    env.reset()
    print("testing")
    print("Current state: {}\n".format(env.state))

    env.step(np.array([2]))
    print("State after applying right action: {0}".format(env.state))
    env.step(np.array([2]))
    print("State after applying right action: {0}".format(env.state))