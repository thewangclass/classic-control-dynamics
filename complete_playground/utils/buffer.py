from collections import deque
import numpy as np
from typing import Union, List, Dict, Any
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

        # Look at sb3/common/preprocessing.py -> get_action_dim
        if(action_type == "Discrete"):
            action_dim = 1
        elif(action_type == "Box"):
            action_dim = int(np.prod(action_space.shape))

        self.actions = np.zeros((self.buffer_size, action_dim), dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminations = np.zeros(self.buffer_size, dtype=np.float32)
        self.timeouts = np.zeros(self.buffer_size, dtype=np.float32)        # infos

        self.pos = 0
        self.full = False
        self.device = get_device(device)

    def add(
            self, 
            state: np.ndarray, 
            next_state: np.ndarray, 
            action: np.ndarray, 
            reward: np.ndarray, 
            terminated: np.ndarray, 
            infos: Dict[str, Any]
    ) -> None:
        
        # sb3, copy to avoid modification by reference
        self.states[self.pos] = np.array(state).copy()
        self.next_states[self.pos] = np.array(next_state).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.terminations[self.pos] = np.array(terminated).copy()
        self.timeouts[self.pos] = np.array(infos.get("TimeLimit.truncated", False)).copy()

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
        # env_indices = np.random.randint(0, high=1, size=(len(batch_inds),))     # high is number of environments, only using 1 for my implementation

        # why normalize observations and rewards? 
        data = (
            self.states[batch_inds, :],                 # [batch_size, state_shape]
            self.actions[batch_inds, :],
            self.next_states[batch_inds, :],
            (self.terminations[batch_inds] * (1 - self.timeouts[batch_inds])).reshape(-1, 1),      # necessary to get tensor into shape [batch, 1]
            self.rewards[batch_inds].reshape(-1, 1)     # necessary to get tensor into shape [batch, 1]
        )
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