
# Get the sign of a value
def get_sign(x):
    if (x == 0): return 0
    else: return x/abs(x)

# # Implement rk4 integration using https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
# # scipy.integrate.RK45
# # Need k1, k2, k3, k4 which represents velocity at current time step, +
# k1 = x_dot                          # change in variable (slope) at current time
# k2 = x_dot + 0.5*self.tau*k1        # slope at midpoint of interval, using y and k1
# k3 = x_dot + 0.5*self.tau*k2        # slope at midpoint of interval, using y and k2
# k4 = x_dot + self.tau*k3            # slope at end of interval
# x = x + self.tau/6 * (k1 + 2*(k2 + k3) + k4)
# Implement rk4 integration, code from https://github.com/chisarie/jax-agents/blob/master/jax_agents/common/runge_kutta.py
def runge_kutta(dynamics_f, state, action, dt):
    """dynamics_f(state, action) -> state_dot."""
    k1 = dynamics_f(state, action)
    k2 = dynamics_f(state + 0.5 * dt * k1, action)
    k3 = dynamics_f(state + 0.5 * dt * k2, action)
    k4 = dynamics_f(state + dt * k3, action)
    next_state = state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return next_state