from ML.networks import Policy
from ML.utils import hard_update
from buffers.PER import PriorityReplayBuffer
from globals import AgentData, Outputs
import torch
from ML.agents.base_agent import Agent
from globals import AgentData, dictionary
import numpy as np
from torch.nn import CrossEntropyLoss


class P_agent(Agent):
    def __init__(self, nA, seed):
        self.dictionary = {i: word.strip() for i, word in enumerate(dictionary)}
        self.nA = nA
        self.seed = seed
        self.gamma = 0.99
        self.gradient_clip = 10
        self.tau = 0.01
        self.criterion = CrossEntropyLoss()

        self.network = Policy(seed, nA)
        self.target_network = Policy(seed, nA)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        # Copy the weights from local to target
        hard_update(self.network, self.target_network)

    def forward(self, state):
        action, prob, probs, value = self.network.forward(state)
        chosen_word = self.dictionary[action.item()]
        return {
            Outputs.ACTION: action,
            Outputs.VALUES: value,
            Outputs.WORD: chosen_word,
            Outputs.ACTION_PROB: prob,
            Outputs.ACTION_PROBS: probs,
        }

    def backward(
        self,
        actions,
        action_prob,
        action_probs,
        states,
        next_states,
        values,
        rewards,
        dones,
        targets,
    ):
        policy_loss = self.criterion(action_probs.squeeze(1), targets.long())
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.soft_update(self.network, self.target_network, self.tau)
