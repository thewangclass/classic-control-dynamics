"""
Code from: https://github.com/chisarie/jax-agents/blob/master/jax_agents/algorithms/random_agent.py
"""
# MIT License

# Copyright (c) 2020 Authors:
#     - Eugenio Chisari <eugenio.chisari@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""A random policy useful for debugging environments."""

from jax import random


class RandomAgent():
    """Generate a random number for each action."""

    def __init__(self, seed, action_dim):
        """Initialize random number gen and action dimension."""
        self.rng = random.PRNGKey(seed)  # rundom number generator
        self.action_dim = action_dim
        self.func = None
        self.state = None
        return

    def select_action(self, *_):
        """Return selected random action."""
        self.rng, rng_input = random.split(self.rng)
        return random.uniform(
            rng_input, shape=(self.action_dim,), minval=-1.0, maxval=1.0)