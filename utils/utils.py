
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



def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Args:
        x: scalar
        m: The lower bound
        M: The upper bound

    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)
