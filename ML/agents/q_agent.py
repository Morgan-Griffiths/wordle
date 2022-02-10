from ML.networks import Q_learning
from ML.utils import hard_update
from buffers.PER import PriorityReplayBuffer
from globals import AgentData, Outputs
import torch
from ML.agents.base_agent import Agent
from globals import AgentData, dictionary
import numpy as np


class Q_agent(Agent):
    def __init__(self, params, config):
        self.dictionary = {i: word.strip() for i, word in enumerate(dictionary)}
        self.nA = params['nA']
        self.seed = config.seed
        self.gamma = config.gamma
        self.gradient_clip = config.gradient_clip
        self.tau = config.tau
        self.eps = config.eps
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.SGD_epoch = config.SGD_epoch
        self.alpha = config.alpha
        self.L2 = config.L2
        self.update_every = config.update_every
        self.learning_update = config.learning_update
        self.device = "cpu"
        self.PER = PriorityReplayBuffer(
            self.buffer_size,
            self.batch_size,
            self.seed,
            alpha=self.alpha,
            device=self.device,
        )
        self.network = Q_learning(self.seed, self.nA)
        self.target_network = Q_learning(self.seed, self.nA)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=1e-4, weight_decay=self.L2
        )
        hard_update(self.network, self.target_network)

    def step_eps(self):
        self.eps = max(self.eps * 0.95, 0)

    def add_memories(self, values, states, next_states, actions, rewards, dones):
        with torch.no_grad():
            for i, state in enumerate(states):
                next_max_action = self.network(next_states[i].unsqueeze(0))[
                    Outputs.VALUES
                ].max(1)[1]
                target = (
                    rewards[i]
                    + self.target_network(next_states[i].unsqueeze(0))[
                        Outputs.VALUES
                    ].squeeze(0)[next_max_action]
                )
                TD_error = target - values[i, actions[i]]
                self.PER.add(
                    state, actions[i], rewards[i], next_states[i], dones[i], TD_error
                )

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
        self.add_memories(values, states, next_states, actions, rewards, dones)
        self.learning_update = (self.learning_update + 1) % self.update_every
        if self.learning_update == 0 and len(self.PER) > self.batch_size:
            for _ in range(self.SGD_epoch):
                samples, indicies, importances = self.PER.sample()
                self.backward(samples, indicies, importances)

    def backward(self, samples, indicies, importances):
        states, actions, rewards, next_states, dones = samples
        with torch.no_grad():
            target_values = self.target_network(next_states)[Outputs.VALUES]
        local_values = self.network(states)[Outputs.VALUES]
        max_target = target_values.gather(1, actions).squeeze(1)
        max_target *= 1 - dones.squeeze(-1)
        targets = rewards + (self.gamma * max_target.unsqueeze(1))
        local = local_values.gather(1, actions)
        TD_errors = local - targets
        loss = ((importances * TD_errors) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()
        TD_errors = np.abs(TD_errors.squeeze(1).detach().cpu().numpy())
        self.PER.sum_tree.update_priorities(TD_errors, indicies)
        self.soft_update(self.network, self.target_network, self.tau)

    def forward(self, state: np.array) -> dict:
        outputs = self.network.forward(state)
        if np.random.random() < self.eps:
            outputs[Outputs.ACTION] = outputs[Outputs.VALUES].max(1)[1]
            outputs[Outputs.WORD] = self.dictionary[outputs[Outputs.ACTION].item()]
        else:
            outputs[Outputs.WORD] = self.dictionary[outputs[Outputs.ACTION].item()]
        return outputs
