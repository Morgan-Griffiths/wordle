import torch
import numpy as np
from globals import AgentData
import os

from utils import to_tensor


class Agent(object):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs: np.array) -> dict:
        state = to_tensor(inputs).unsqueeze(0)
        return self.forward(state)

    def learn(self, player_data):
        action_probs = player_data[AgentData.ACTION_PROBS]
        action_prob = player_data[AgentData.ACTION_PROB]
        values = player_data[AgentData.VALUES]
        rewards = player_data[AgentData.REWARDS]
        actions = player_data[AgentData.ACTIONS]
        states = player_data[AgentData.STATES]
        dones = player_data[AgentData.DONES]
        targets = player_data[AgentData.TARGETS]
        dones = torch.stack(dones)
        rewards = torch.stack(rewards)
        next_states = torch.stack(states[1:])
        states = torch.stack(states[:-1])
        actions = torch.stack(actions)
        action_probs = torch.stack(action_probs)
        action_prob = torch.stack(action_prob)
        values = torch.stack(values).squeeze(1)
        targets = torch.stack(targets[:-1])
        self.backward(
            actions,
            action_prob,
            action_probs,
            states,
            next_states,
            values,
            rewards,
            dones,
            targets,
        )

    def load_weights(self, path):
        self.network.load_state_dict(torch.load(path))
        self.network.eval()

    def save_weights(self, path):
        # print(f"saving weights to {path}")
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.network.state_dict(), path)

    def update_networks(self):
        self.target_critic = Agent.soft_update_target(
            self.local_critic, self.target_critic, self.tau
        )
        self.target_actor = Agent.soft_update_target(
            self.local_actor, self.target_actor, self.tau
        )

    def evaluate(self):
        self.network.eval()

    def train(self):
        self.network.train()

    @staticmethod
    def soft_update(local, target, tau):
        for local_param, target_param in zip(local.parameters(), target.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1 - tau) * target_param.data
            )
