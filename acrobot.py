"""
Classic cart-pole system implemented by Rich Sutton et al.
Framework from: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/acrobot.py
"""

import math
import numpy as np

from numpy import cos, pi, sin
from utils import get_sign
from utils import runge_kutta as rk4



__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py


class Acrobot():
    """
    ## Description

    The Acrobot environment is based on Sutton's work in
    ["Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html)
    and [Sutton and Barto's book](http://www.incompleteideas.net/book/the-book-2nd.html).
    The system consists of two links connected linearly to form a chain, with one end of
    the chain fixed. The joint between the two links is actuated. The goal is to apply
    torques on the actuated joint to swing the free end of the linear chain above a
    given height while starting from the initial state of hanging downwards.

    As seen in the **Gif**: two blue links connected by two green joints. The joint in
    between the two links is actuated. The goal is to swing the free end of the outer-link
    to reach the target height (black horizontal line above system) by applying torque on
    the actuator.

    ## Action Space

    The action is discrete, deterministic, and represents the torque applied on the actuated
    joint between the two links.

    | Num | Action                                | Unit         |
    |-----|---------------------------------------|--------------|
    | 0   | apply -1 torque to the actuated joint | torque (N m) |
    | 1   | apply 0 torque to the actuated joint  | torque (N m) |
    | 2   | apply 1 torque to the actuated joint  | torque (N m) |

    ## Observation Space

    The observation is a `ndarray` with shape `(6,)` that provides information about the
    two rotational joint angles as well as their angular velocities:

    | Num | Observation                  | Min                 | Max               |
    |-----|------------------------------|---------------------|-------------------|
    | 0   | Cosine of `theta1`           | -1                  | 1                 |
    | 1   | Sine of `theta1`             | -1                  | 1                 |
    | 2   | Cosine of `theta2`           | -1                  | 1                 |
    | 3   | Sine of `theta2`             | -1                  | 1                 |
    | 4   | Angular velocity of `theta1` | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
    | 5   | Angular velocity of `theta2` | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |

    where
    - `theta1` is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly
    downwards.
    - `theta2` is ***relative to the angle of the first link.***
        An angle of 0 corresponds to having the same angle between the two links.

    The angular velocities of `theta1` and `theta2` are bounded at ±4π, and ±9π rad/s respectively.
    A state of `[1, 0, 1, 0, ..., ...]` indicates that both links are pointing downwards.

    ## Rewards

    The goal is to have the free end reach a designated target height in as few steps as possible,
    and as such all steps that do not reach the goal incur a reward of -1.
    Achieving the target height results in termination with a reward of 0. The reward threshold is -100.

    ## Starting State

    Each parameter in the underlying state (`theta1`, `theta2`, and the two angular velocities) is initialized
    uniformly between -0.1 and 0.1. This means both links are pointing downwards with some initial stochasticity.

    ## Episode End

    The episode ends if one of the following occurs:
    1. Termination: The free end reaches the target height, which is constructed as:
    `-cos(theta1) - cos(theta2 + theta1) > 1.0`
    2. Truncation: Episode length is greater than 500 (200 for v0)
    pass
"""

    def __init__(self):
        self.link_length_1 = 1.0    # [m]
        self.link_length_2 = 1.0    # [m]
        self.link_mass_1 = 1.0      # [kg] mass of link 1
        self.link_mass_2 = 1.0      # [kg] mass of link 2
        self.link_com_pos_1 = 0.5   # [m] position of the center of mass of link 1
        self.link_com_pos_2 = 0.5   # [m] position of the center of mass of link 2
        self.link_moi = 1.0         # moment of inertia for both links

        self.max_vel_1 = 4 * pi     # values taken from gymnasium acrobot
        self.max_vel_2 = 9 * pi

        self.avail_torque = [-1.0, 0.0, +1]
        self.torque_noise_max = 0.0

        # Possible actions the acrobot can take
        # 0: apply -1 torque to the actuated joint
        # 1: apply 0 torque to the actuated joint
        # 2: apply 1 torque to the actuated joint
        self.action_space = {0, 1, 2}

        # Consider renaming these variables as bounds
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.max_vel_1, self.max_vel_2], dtype=np.float32
        )
        low = -high

        self.state = None

    def reset(self):
        """
        Reset initial state for next episode.
        High is given an initial default value taken from gymnasium acrobot.py. This is to provide some variation in the starting state and make the model learn a more generalizable policy.
        """
        high = 0.1
        low = -high

        theta1, theta2, theta1_dot, theta2_dot = np.random.uniform(low=low, high=high, size=(4,)).astype(np.float32)
        self.state = np.array(
            [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot], dtype=np.float32     
        )

        return self.state
    

    def step(self, action):
        # Make sure valid action and state are present
        assert action in self.action_space, f"invalid action chosen: {action}"
        assert self.state is not None, "Call reset before step"
        
        torque = self.avail_torque[action]  # -1, 0, 1



if __name__ == '__main__':
    abot = Acrobot()
    abot.reset()
    print("testing")
    print("Current state: {}".format(abot.state))