from collections import deque
import numpy as np
from typing import Union
import torch as th
from complete_playground.utils.common import get_device
from complete_playground.utils.type_aliases import ReplayBufferSamples


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
        self.states = np.zeros(shape=(self.buffer_size, *observation_space.shape), dtype=observation_space.dtype)
        self.next_states = np.zeros((self.buffer_size, *observation_space.shape), dtype=observation_space.dtype)
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
            done,
            info):
        
        # sb3, copy to avoid modification by reference
        self.states[self.pos] = np.array(state).copy()
        self.next_states[self.pos] = np.array(next_state).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.terminations[self.pos] = np.array(terminated).copy()
        self.truncations[self.pos] = np.array(truncated).copy()
        self.dones[self.pos] = np.array(done).copy()

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

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)
    
    def _get_samples(self, batch_inds: np.ndarray):
        env_indices = np.random.randint(0, size=(len(batch_inds),))

        # why normalize observations and rewards? 
        data = {
            self.states[batch_inds, :],
            self.actions[batch_inds, :],
            self.next_states[batch_inds, :],
            self.dones[batch_inds, :],
            self.rewards[batch_inds, :]
        }

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
    
    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)