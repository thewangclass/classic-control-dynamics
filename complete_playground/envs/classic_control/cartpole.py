"""
Classic cart-pole system implemented by Rich Sutton et al.
Framework from: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
Dynamics from: https://coneural.org/florian/papers/05_cart_pole.pdf

"""
import sys
sys.path.append("/home/thewangclass/projects/classic-control-dynamics/")
import math
import numpy as np

from complete_playground.envs.utils import get_sign
from complete_playground.envs.utils import runge_kutta as rk4
from complete_playground import Env, spaces


class CartPoleEnv(Env):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

    Our implementation captures the full dynamical system as described in ["Correct equations for the dynamics of the cart-pole system](https://coneural.org/florian/papers/05_cart_pole.pdf). Notably, it allows for the possibility of accounting for frictional forces.

    ## Action Space

    The action is a python set which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it.

    ## Observation Space
    The bounds observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    
    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500
    """


    def __init__(self) -> None:
        ##################################################
        # EPISODE ENDING
        ##################################################
        self.steps = 0

        # Angle at which to fail the episode: pole below this angle means failure
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # Left/right bounds of space: leaving these boundaries means failure
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        # Really these should be called upper and lower bounds.
        self.upper_bound = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,    # np.inf
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,     # np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.lower_bound = -self.upper_bound

        ##################################################
        # SYSTEM DIMENSIONS
        ##################################################
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.mass_total = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length, assumed to be 1
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0   # magnitude of the force applied

        # friction
        self.mu_cart = 0    # coeff friction of cart
        self.mu_pole = 0    # coeff friction of pole
        self.force_normal_cart = 1

        # acceleration calculations
        self.theta_acc = None
        self.x_acc = None

        ##################################################
        # CALCULATION OPTIONS
        ##################################################
        # metadata lists "render_fps" as 50. This is where the tau value of 0.02 comes from because 50 frames per second results in 1/50 which is 0.02 seconds per frame.
        self.tau = 0.02  # seconds between state updates, our delta_t
        self.kinematics_integrator = "rk4"  # we use rk4 for our integration

        ##################################################
        # DEFINE ACTION AND OBSERVATION SPACE
        ##################################################
        # Possible actions the cartpole can take
        # 0: push cart to left
        # 1: push cart to right
        # self.action_space = {0, 1}
        # self.action_type = "Discrete"  # used in buffer to determine shape of memory
        # self.observation_space = self.upper_bound # to use in network for first layer input
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(self.lower_bound, self.upper_bound, dtype=np.float32)
        self.state = None

        ##################################################
        # INITIALIZE EPISODIC REWARD AND LENGTH
        ##################################################
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self):
        """
        Reset initial state for next session.
        High is given an initial default value. Default value is taken from gymnasium cartpole.py. This is to provide some variation in the starting state and make the model learn a more generalizable policy.
        """
        ##################################################
        # INITIALIZE NEW RANDOM STARTING STATE
        ##################################################
        high = 0.01    
        low = -high    
        self.state = np.random.uniform(low=low, high=high, size=(4,))

        ##################################################
        # RESET EPISODIC VALUES
        ##################################################   
        self.steps = 0
        self.episode_reward = 0
        self.episode_length = 0

        return np.array([self.state], dtype=np.float32)


    def step(self, action):
        assert self.action_space.contains(
            action[0]
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before step"

        # update state
        force = self.force_mag if action == 1 else -self.force_mag
        self.calc_x_acc(force)  # calc_x_acc updates theta_acc first to be used in x_acc calculation
        self.state = rk4(self.dynamics_cartpole, self.state, force, self.tau)

        # check if episode ends due to termination and update reward accordingly
        terminated = self.check_termination(self.state)
        reward = 1.0 if not terminated else 0.0

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
        
        # next_state, reward, terminated, truncated, info
        return np.array([self.state], dtype=np.float32), reward, terminated, truncated, info

    def check_termination(self, state):
        x = state[0]
        theta = state[2]
        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

    def dynamics_cartpole(self, current_state, action):
        # current state is comprised of x, x_dot, theta, theta_dot
        # change in each of these is x_dot, x_acc, theta_dot, theta_acc
        x, x_dot, theta, theta_dot = current_state
        return np.array([x_dot, self.x_acc, theta_dot, self.theta_acc], dtype=np.float32)
    
    def calc_theta_acc(self, force):
        # Get position, velocity, angle, and angular velocity from state
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Calculate new theta_acc, then normal force of cart, then x_acc according to https://coneural.org/florian/papers/05_cart_pole.pdf
        force_normal_cart = self.force_normal_cart

        theta_acc = (self.gravity*sintheta * costheta*(
                            (
                                (-force - self.masspole*self.length*theta_dot**2*(
                                sintheta + self.mu_cart*get_sign(force_normal_cart*x_dot)*costheta
                                )) / self.mass_total    
                            ) + self.mu_cart*self.gravity*get_sign(force_normal_cart*x_dot)
                        ) - (self.mu_pole*theta_dot)/(self.masspole*self.length)
                    )

        theta_acc = theta_acc / (self.length * 
                                    (4.0/3.0 - (self.masspole*costheta/self.mass_total)*
                                        (costheta - self.mu_cart*get_sign(force_normal_cart*x_dot))
                                    )
                                )
        
        force_normal_cart = self.mass_total*self.gravity - (self.masspole * self.length * (theta_acc * sintheta + theta_acc**2 * costheta))

        # if normal force changes sign, then compute theta_acc taking into account the new sign
        if get_sign(force_normal_cart) != get_sign(self.force_normal_cart):
            self.force_normal_cart = force_normal_cart      # update sign of normal force

            theta_acc = (self.gravity*sintheta * costheta*(
            (
                (-force - self.masspole*self.length*theta_dot**22*(
                sintheta + self.mu_cart*get_sign(force_normal_cart*x_dot)*costheta
                )) / self.mass_total    
            ) + self.mu_cart*self.gravity*get_sign(force_normal_cart*x_dot)
            ) - (self.mu_pole*theta_dot)/(self.masspole*self.length))

            theta_acc = theta_acc / (self.length * 
                                    (4.0/3.0 - (self.masspole*costheta/self.mass_total)*
                                        (costheta - self.mu_cart*get_sign(force_normal_cart*x_dot))
                                    )
                                )
        self.theta_acc = theta_acc
        return theta_acc

    def calc_x_acc(self, force):
        self.calc_theta_acc(force)

        # Get position, velocity, angle, and angular velocity from state
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        x_acc = (force + self.masspole * self.length * (theta_dot**2 * sintheta - self.theta_acc * costheta) - (self.mu_cart * self.force_normal_cart * get_sign(self.force_normal_cart * x_dot))) / self.mass_total

        self.x_acc = x_acc
        return x_acc