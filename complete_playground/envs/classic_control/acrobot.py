"""
Classic cart-pole system implemented by Rich Sutton et al.
Framework from: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/acrobot.py
"""
import numpy as np
from numpy import cos, pi, sin
from complete_playground.envs.utils import wrap, bound
from complete_playground import Env, spaces
from complete_playground.envs.utils import runge_kutta as rk4

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


class AcrobotEnv(Env):
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

    The observation is a `ndarray` with shape `(4,)` that provides information about the two rotational joint angles as well as their angular velocities:

    | Num | Observation                  | Min                 | Max               |
    |-----|------------------------------|---------------------|-------------------|
    | 0   | `theta1`                     | -pi                 | pi                |
    | 1   | `theta2`                     | -pi                 | pi                |
    | 2   | Angular velocity of `theta1` | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
    | 3   | Angular velocity of `theta2` | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |

    where
    - `theta1` is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly
    downwards.
    - `theta2` is ***relative to the angle of the first link.***
    An angle of 0 corresponds to having the same angle between the two links.

    The angular velocities of `theta1` and `theta2` are bounded at ±4π, and ±9π rad/s respectively.
    
    Note: we follow https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py when defining the state space, not the one in gymnasium where they use a (6,) ndarray.

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
        ##################################################
        # EPISODE ENDING
        ##################################################
        self.steps = 0

        # constraints
        self.max_vel_1 = 4 * pi     # values taken from gymnasium acrobot
        self.max_vel_2 = 9 * pi
        self.upper_bound = np.array(
            [1.0, 1.0, 1.0, 1.0, self.max_vel_1, self.max_vel_2], dtype=np.float32
        )
        self.lower_bound = -self.upper_bound

        ##################################################
        # SYSTEM DIMENSIONS
        ##################################################
        self.gravity = 9.8
        self.link_length_1 = 1.0    # [m]
        self.link_length_2 = 1.0    # [m]
        self.link_mass_1 = 1.0      # [kg] mass of link 1
        self.link_mass_2 = 1.0      # [kg] mass of link 2
        self.link_com_pos_1 = 0.5   # [m] position of the center of mass of link 1
        self.link_com_pos_2 = 0.5   # [m] position of the center of mass of link 2
        self.link_moi = 1.0         # moment of inertia for both links

        self.avail_torque = np.array([-1.0, 0.0, +1])
        self.torque_noise_max = 0.0

        ##################################################
        # CALCULATION OPTIONS
        ##################################################
        # metadata lists "render_fps" as 50. This is where the tau value of 0.02 comes from because 50 frames per second results in 1/50 which is 0.02 seconds per frame.
        self.tau = 0.02  # seconds between state updates, our delta_t
        self.kinematics_integrator = "rk4"  # we use rk4 for our integration

        ##################################################
        # DEFINE ACTION AND OBSERVATION SPACE
        ##################################################
        # Possible actions the acrobot can take
        # 0: apply -1 torque to the actuated joint
        # 1: apply 0 torque to the actuated joint
        # 2: apply 1 torque to the actuated joint
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=self.lower_bound, high=self.upper_bound, dtype=np.float32)
        self.state = None

        ##################################################
        # INITIALIZE EPISODIC REWARD AND LENGTH
        ##################################################
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self):
        """
        Reset initial state for next episode.
        High is given an initial default value taken from gymnasium acrobot.py. This is to provide some variation in the starting state and make the model learn a more generalizable policy.
        """
        ##################################################
        # INITIALIZE NEW RANDOM STARTING STATE
        ##################################################
        high = 0.1
        low = -high
        self.state = np.random.uniform(low=low, high=high, size=(4,)).astype(np.float32)

        ##################################################
        # RESET EPISODIC VALUES
        ##################################################   
        self.steps = 0
        self.episode_reward = 0
        self.episode_length = 0

        return np.array([self._get_ob()])
    
    
    def step(self, action):
        # Make sure valid action and state are present
        assert self.action_space.contains(
            action[0]
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before step"

        # update state
        torque = self.avail_torque[action]  # torque is determined by the action chosen: -1, 0, 1
        if self.torque_noise_max > 0:       # add random noise to torque
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max
            )        
        ns = rk4(self.dynamics_acrobot, self.state, torque, self.tau)
        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.max_vel_1, self.max_vel_1)
        ns[3] = bound(ns[3], -self.max_vel_2, self.max_vel_2)
        self.state = ns

        # check if episode ends due to termination and update reward accordingly
        terminated = self.check_termination(self.state)
        reward = -1.0 if not terminated else 0.0

        # check truncation
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps

        # generate infos for last timestep of episode
        self.episode_reward += reward
        self.episode_length += 1
        info = {}
        if terminated or truncated:
            info['final_observation'] = self._get_ob()
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

        return np.array([self._get_ob()]), reward, terminated, truncated, info
    
    def dynamics_acrobot(self, current_state, action):
        m1 = self.link_mass_1
        m2 = self.link_mass_2
        l1 = self.link_length_1
        lc1 = self.link_com_pos_1
        lc2 = self.link_com_pos_2
        I1 = self.link_moi
        I2 = self.link_moi
        g = self.gravity
        a = action
        s = current_state

        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        # http://incompleteideas.net/book/11/node4.html
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )
        ddtheta2 = (
            a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
        ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2], dtype=np.float32)

    
    def check_termination(self, state):
        assert state is not None, "Call reset before using AcrobotEnv object."
        return bool(-cos(state[0]) - cos(state[1] + state[0]) > 1.0)
    
    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
    )

if __name__ == '__main__':
    abot = AcrobotEnv()
    abot.reset()
    print("testing")
    print("Current state: {}\n".format(abot.state))

    t=2
    abot.step(t)
    print("State after applying  {0}Nm torque for {1}seconds to the actuated joint: {2}\n".format(abot.avail_torque[t], abot.tau, abot.state))
