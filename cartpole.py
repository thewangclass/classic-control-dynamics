"""
Classic cart-pole system implemented by Rich Sutton et al.
Used from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py and https://perma.cc/C9ZM-652R

"""
import math
import numpy as np

from utils import get_sign
from utils import runge_kutta as rk4

class CartPole():
    """
    
    """


    def __init__(self) -> None:
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


        # metadata lists "render_fps" as 50. This is where the tau value of 0.02 comes from because 50 frames per second results in 1/50 which is 0.02 seconds per frame.
        self.tau = 0.02  # seconds between state updates, our delta_t



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
        """
        Reset initial state for next session.
        High is given an initial default value. Default value is taken from gymnasium cartpole.py. This is to provide some variation in the starting state and make the model learn a more generalizable policy.
        """
        high = 0.05    
        low = -high     
        self.state = np.random.uniform(low=low, high=high, size=(4,)).astype(np.float32)

        return np.array(self.state, dtype=np.float32)


    def step(self, action):
        # Make sure valid action and state are present
        assert action in self.action_space, f"invalid action chosen: {action}"
        assert self.state is not None, "Call reset before step"

        dynamics_function = self.dynamics(action)

        # # Implement rk4 integration using https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        # # scipy.integrate.RK45
        # # Need k1, k2, k3, k4 which represents velocity at current time step, +
        # k1 = x_dot                          # change in variable (slope) at current time
        # k2 = x_dot + 0.5*self.tau*k1        # slope at midpoint of interval, using y and k1
        # k3 = x_dot + 0.5*self.tau*k2        # slope at midpoint of interval, using y and k2
        # k4 = x_dot + self.tau*k3            # slope at end of interval
        # x = x + self.tau/6 * (k1 + 2*(k2 + k3) + k4)

    
    def dynamics(self, action):
         # Get position, velocity, angle, and angular velocity from state
        x, x_dot, theta, theta_dot = self.state

        print(self.state)
        print(type(self.state))

        # If action is 1, then move right, else we use negative force to the left
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Calculate theta_acceleration and x_acceleration
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.mass_total

        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.mass_total)
        )
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.mass_total


        # Calculate new theta_acc, then normal force of cart, then x_acc according to https://coneural.org/florian/papers/05_cart_pole.pdf
        force_normal_cart = self.force_normal_cart

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
        
        force_normal_cart = self.mass_total*self.gravity - (self.masspole * self.length * (theta_acc * sintheta + theta_acc**2 * costheta))

        # if normal force changes sign, then compute theta_acc taking into account the new sign
        if get_sign(force_normal_cart) != get_sign(self.force_normal_cart):
            self.force_normal_cart = force_normal_cart  # update sign of normal force
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

        x_acc = (force + self.masspole * self.length * (theta_dot**2 * sintheta - theta_acc * costheta) - (self.mu_cart * force_normal_cart * get_sign(force_normal_cart * x_dot))) / self.mass_total








    
             

if __name__ == '__main__':
    cart = CartPole()
    print(cart.state)
    print(cart.reset())
    cart.step(1)