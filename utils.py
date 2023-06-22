
# Get the sign of a value
def get_sign(x):
    if (x == 0): return 0
    else: return x/abs(x)


# Implement rk4 integration, code from https://github.com/chisarie/jax-agents/blob/master/jax_agents/common/runge_kutta.py
def runge_kutta(dynamics_f, state, action, dt):
    """dynamics_f(state, action) -> state_dot."""
    k1 = dynamics_f(state, action)
    k2 = dynamics_f(state + 0.5 * dt * k1, action)
    k3 = dynamics_f(state + 0.5 * dt * k2, action)
    k4 = dynamics_f(state + dt * k3, action)
    update = (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return update