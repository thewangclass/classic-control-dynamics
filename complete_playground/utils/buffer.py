from collections import deque
import numpy as np

class ReplayBuffer():
    """
    Class that represents a Replay Buffer
    """
    def __init__(self,max_len):
        self.max_len = max_len
        self.state = deque(maxlen = max_len)
        self.next_state = deque(maxlen=max_len)
        self.action = deque(maxlen = max_len)
        self.reward = deque(maxlen = max_len)
        self.terminated = deque(maxlen=max_len)
        self.truncated = deque(maxlen=max_len)
        self.info = deque(maxlen = max_len)

    def add(self, state, next_state, action, reward, terminated, truncated, info):
        self.state.append(state)
        self.next_state.append(next_state)
        self.action.append(action)
        self.reward.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.info.append(info)

    def sample_transitions(self, batch_size):
        batch = np.random.permutation(len(self.frames))[:batch_size]
        trans = np.array(len(self.frames))[batch]
        return trans    