from collections import deque
import hyp
import random
import collections


class Result(
    collections.namedtuple('Result', ['state', 'reward', 'terminated'])):
  """A namedtuple defines the result of a step for the molecule class.

    The namedtuple contains the following fields:
      state: Chem.RWMol. The molecule reached after taking the action.
      reward: Float. The reward get after taking the action.
      terminated: Boolean. Whether this episode is terminated.
  """


class ReplayMemory(object):

    def __init__(self):
        self.max_size = hyp.replay_buffer_size
        self.memory = deque([],maxlen=hyp.replay_buffer_size)

    def push(self, x):
        """Saves a Result"""
        self.memory.append(x)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        (state_t, reward_t, state_tp1, done_mask) = map(np.stack, zip(*batch))
        return state_t, reward_t, state_tp1, done_mask

    def __len__(self):
        return len(self.memory)
