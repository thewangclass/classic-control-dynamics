"""
Classic cart-pole system implemented by Rich Sutton et al.
Used from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py and https://perma.cc/C9ZM-652R

"""
import math
import numpy as np

from typing import Sequence

class CartPole():
    """
    
    """


    def __init__(self) -> None:
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.mass_total = self.mass_pole + self.mass_cart
        self.length = 0.5  # actually half the pole's length, assumed to be 1
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "rk4"  # we use rk4 for our integration

        # Angle at which to fail the episode: pole below this angle means failure
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # Left/right bounds of space: leaving these boundaries means failure
        self.x_threshold = 2.4

        

        # Possible actions the cartpole can take
        # 0 push cart to left, 1 push cart to right
        self.action_space = {0, 1}

        """
        The bounds observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

        | Num | Observation           | Min                 | Max               |
        |-----|-----------------------|---------------------|-------------------|
        | 0   | Cart Position         | -4.8                | 4.8               |
        | 1   | Cart Velocity         | -Inf                | Inf               |
        | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
        | 3   | Pole Angular Velocity | -Inf                | Inf               |
        """
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,    # np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.inf,     # np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        low = -high

        # CartPole represented by (CartPosition, CartVelocity, PoleAngle, PoleAngVelocity)
        # Starting state is initialized randomly in reset()
        self.state = None

    def reset(self):
        high = 0.05     # default high
        low = -high     # defaults taken from gymnasium cartpole.py
        self.state = np.random.uniform(low=low, high=high, size=(4,)).astype(np.float32)

        return np.array(self.state, dtype=np.float32)


    def step(self):
        pass
    
             

if __name__ == '__main__':
    cart = CartPole()
    print(cart.state)
    print(cart.reset())