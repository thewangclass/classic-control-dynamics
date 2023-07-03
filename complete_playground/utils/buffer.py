from collections import deque
import numpy as np
from typing import Union
import torch as th
from complete_playground.utils.common import get_device


class ReplayBuffer():
    """
    Class that represents a Replay Buffer
    """
    def __init__(
            self, 
            buffer_size: int,
            observation_space,
            action_space,
            device: Union[th.device, str] = "auto"
        ):
        self.buffer_size = buffer_size
        self.states = np.zeros((self.buffer_size, observation_space.shape), dtype=observation_space.dtype)
        self.next_states = np.zeros((self.buffer_size, observation_space.shape), dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, len(action_space)), dtype=np.int32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminations = np.zeros(self.buffer_size, dtype=np.float32)
        self.truncations = np.zeros(self.buffer_size, dtype=np.float32)
        # self.info = deque(maxlen = max_len)

        self.pos = 0
        self.full = False
        self.device = get_device(device)

    def add(
            self, 
            state, 
            next_state, 
            action, 
            reward, 
            terminated, 
            truncated, 
            info):
        
        # sb3, copy to avoid modification by reference
        self.states[self.pos] = np.array(state).copy()
        self.next_states[self.pos] = np.array(next_state).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.truncations[self.pos] = np.array(terminated).copy()
        self.truncations[self.pos] = np.array(truncated).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample_transitions(self, batch_size):
        batch = np.random.permutation(len(self.frames))[:batch_size]
        trans = np.array(len(self.frames))[batch]
        return trans    