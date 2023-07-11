"""
Code originally from Gymnasium. Modified by me. 
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import sys
sys.path.append("/home/thewangclass/projects/classic-control-dynamics/")
import math
from typing import Optional

import numpy as np
from complete_playground import spaces

class MountainCarEnv():
    """
    ## Description

    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
    This version is the one with discrete actions.

    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```

    ## Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                          | Min   | Max  | Unit         |
    |-----|--------------------------------------|-------|------|--------------|
    | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m) |
    | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v) |

    ## Action Space

    There are 3 discrete deterministic actions:

    - 0: Accelerate to the left
    - 1: Don't accelerate
    - 2: Accelerate to the right

    ## Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0
    upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and
    velocity is clipped to the range `[-0.07, 0.07]`.

    ## Reward:

    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
    penalised with a reward of -1 for each timestep.

    ## Starting State

    The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
    The starting velocity of the car is always assigned to 0.

    ## Episode End

    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 200.


    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('MountainCar-v0')
    ```

    ## Version History

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
        self.max_episode_steps = 200
        self.steps = 0

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0

        self.lower_bound = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.upper_bound = np.array([self.max_position, self.max_speed], dtype=np.float32)

        ##################################################
        # SYSTEM DIMENSIONS
        ##################################################
        self.force = 0.001
        self.gravity = 0.0025

        ##################################################
        # DEFINE ACTION AND OBSERVATION SPACE
        ##################################################
        # self.action_space = {0, 1, 2}
        # self.action_type = "Discrete"
        # self.observation_space = self.upper_bound
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.lower_bound, self.upper_bound, dtype=np.float32)
        self.state = None

        ##################################################
        # INITIALIZE EPISODIC REWARD AND LENGTH
        ##################################################
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self):
        ##################################################
        # INITIALIZE NEW RANDOM STARTING STATE
        ##################################################
        low = -0.6
        high = -0.4
        self.state = np.random.uniform(low=low, high=high, size=(2,)).astype(np.float32)


        ##################################################
        # RESET EPISODIC VALUES
        ##################################################   
        self.steps = 0
        self.episode_reward = 0
        self.episode_length = 0

        return np.array([self.state])

    def step(self, action: int):
        # assert self.action_space.contains(
        #     action
        # ), f"{action!r} ({type(action)}) invalid"

        # update state
        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = np.array([0])
        self.state = np.array([position, velocity]).flatten()   # different

        # check if episode ends due to termination and update reward accordingly
        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0

        # check truncation (episode ending due to time limit or some other reason not defined as part of the task MDP) 
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

        return np.array([self.state], dtype=np.float32), reward, terminated, truncated, info

if __name__ == '__main__':
    env = MountainCar()
    env.reset()
    print("testing")
    print("Current state: {}\n".format(env.state))

    env.step(2)
    print("State after applying right action: {0}".format(env.state))
    env.step(2)
    print("State after applying right action: {0}".format(env.state))
    # env.step(1)
    # print("State after applying no action: {0}".format(env.state))
    # env.step(1)
    # print("State after applying no action: {0}".format(env.state))
    # env.step(0)
    # print("State after applying left action: {0}".format(env.state))
    # env.step(0)
    # print("State after applying left action: {0}".format(env.state))
