import numpy as np
import random
import torch
from collections import namedtuple
from buffers.priority_tree import PriorityTree

"""
Priority Experience Replay
alpha(priority or w) dictates how biased the sampling should be towards the TD error. 0 < a < 1
beta(IS) informs the importance of the sample update
We're increasing beta(IS) from 0.5 to 1 over time
alpha(priority) we're holding constant at 0.5
"""


class PriorityReplayBuffer(object):
    def __init__(
        self,
        buffer_size,
        batch_size,
        seed,
        alpha=0.5,
        beta=0.5,
        beta_end=1,
        beta_duration=1e4,
        epsilon=7e-5,
        device=None,
    ):

        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_end = beta_end
        self.beta_duration = beta_duration
        self.beta_increment = (beta_end - beta) / beta_duration
        self.max_w = 0
        self.epsilon = epsilon
        self.TD_sum = 0
        self.index = 0
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.experience = namedtuple(
            "experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.sum_tree = PriorityTree(buffer_size, batch_size, alpha, epsilon)
        self.memory = {}

    def add(self, state, action, reward, next_state, done, TD_error):
        e = self.experience(state, action, reward, next_state, done)
        # add memory to memory and add corresponding priority to the priority tree
        self.memory[self.index] = e
        self.sum_tree.add(TD_error, self.index)
        self.index = (self.index + 1) % self.buffer_size

    def sample(self):
        # We times the error by these weights for the updates
        # Or we could sum once on the first sample and then keep track of what we add and lose from the buffer.
        # priority^a over the sum of the priorities^a = likelyhood of the given choice
        # Anneal beta
        self.update_beta()
        # Get the samples and indicies
        priorities, indicies = self.sum_tree.sample(self.index, len(self))
        # Normalize with the sum
        norm_priorities = priorities / self.sum_tree.root.value
        samples = [self.memory[index] for index in indicies]
        # Importance weights
        importances = np.stack(
            [
                (priority * self.buffer_size) ** -self.beta
                for priority in norm_priorities
            ]
        )
        self.max_w = max(self.max_w, max(importances))
        # Normalize importance weights
        norm_importances = torch.tensor(importances / self.max_w)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.from_numpy(np.vstack(dones)).to(self.device)

        # states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).float().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(int)).float().to(self.device)

        return (
            (states, actions, rewards, next_states, dones),
            indicies,
            norm_importances,
        )

    def update_beta(self):
        self.beta += self.beta_increment
        self.beta = round(min(self.beta, self.beta_end), 3)

    def __len__(self):
        return len(self.memory.keys())
