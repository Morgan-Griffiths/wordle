from lib2to3.refactor import MultiprocessingUnsupported
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch import optim
from ML.networks import Policy, Q_learning
from globals import AgentData, Outputs, dictionary, Models
import os

from ML.utils import hard_update
from utils import to_tensor


class Agent(object):
    def __init__(self, nA, seed, model):
        super().__init__()
        self.dictionary = {i: word.strip() for i, word in enumerate(dictionary)}
        self.nA = nA
        self.model = model
        self.seed = seed
        self.gamma = 0.99
        self.gradient_clip = 10
        self.tau = 0.01
        if model == Models.Q_LEARNING:
            self.network = Q_learning(seed, nA)
            self.target_network = Q_learning(seed, nA)
            self.backward = self.backward_q
            self.forward = self.q_forward
        elif model == Models.AC_LEARNING:
            self.network = Policy(seed, nA)
            self.target_network = Policy(seed, nA)
            self.backward = self.backward_ac
            self.forward = self.ac_forward
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        # Copy the weights from local to target
        hard_update(self.network, self.target_network)

    def ac_forward(self, state):
        action, prob, probs, value = self.network.forward(state)
        chosen_word = self.dictionary[action.item()]
        return {
            Outputs.ACTION: action,
            Outputs.VALUES: value,
            Outputs.WORD: chosen_word,
            Outputs.ACTION_PROB: prob,
            Outputs.ACTION_PROBS: probs,
        }

    def q_forward(self, state: np.array) -> dict:
        outputs = self.network.forward(state)
        outputs[Outputs.WORD] = self.dictionary[outputs[Outputs.ACTION].item()]
        return outputs

    def __call__(self, inputs: np.array) -> dict:
        state = to_tensor(inputs).unsqueeze(0)
        return self.forward(state)

    def learn(self, player_data):
        action_probs = player_data[AgentData.ACTION_PROBS]
        values = player_data[AgentData.VALUES]
        rewards = player_data[AgentData.REWARDS]
        actions = player_data[AgentData.ACTIONS]
        states = player_data[AgentData.STATES]
        dones = player_data[AgentData.DONES]
        dones = torch.stack(dones)
        rewards = torch.stack(rewards)
        next_states = torch.stack(states[1:])
        states = torch.stack(states[:-1])
        actions = torch.stack(actions)
        action_probs = torch.stack(action_probs)
        values = torch.stack(values).squeeze(1)
        self.backward(
            actions, action_probs, states, next_states, values, rewards, dones
        )

    def backward_q(
        self, actions, action_probs, states, next_states, values, rewards, dones
    ):
        # rows = torch.arange(values.shape[0])
        # local_outcomes = self.network(next_states)[]
        # local_next_state_actions = local_next_state_values.max(1)[1].unsqueeze(1)
        with torch.no_grad():
            target_values = self.target_network(next_states)[Outputs.VALUES]
        max_target = target_values.gather(1, actions).squeeze(1)
        max_target *= 1 - dones[1:]
        targets = rewards + (self.gamma * max_target.unsqueeze(1))
        local = values.gather(1, actions)
        TD_error = F.smooth_l1_loss(local, targets, reduction="sum")
        self.optimizer.zero_grad()
        TD_error.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.soft_update(self.network, self.target_network, self.tau)

    def backward_ac(
        self, actions, action_probs, states, next_states, values, rewards, dones
    ):
        _, _, _, target_values = self.target_network(states)
        rows = torch.arange(values.shape[0])
        action_values = target_values[rows, actions.squeeze(-1)].unsqueeze(-1)
        policy_loss = (-action_probs.view(-1) * action_values).sum()
        critic_loss = F.smooth_l1_loss(
            rewards.squeeze(-1), values[rows, actions.squeeze(-1)], reduction="sum"
        )
        loss = policy_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.soft_update(self.network, self.target_network, self.tau)

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
