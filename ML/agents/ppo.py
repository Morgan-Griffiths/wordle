from ML.networks import PPO
from globals import AgentData, Outputs
import torch
from torch import optim
from ML.agents.base_agent import Agent
from globals import AgentData, dictionary
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from utils import to_tensor, count_parameters


class PPO_agent(Agent):
    def __init__(self, params, config):
        self.dictionary = {i: word.strip() for i, word in enumerate(dictionary)}
        self.nA = params["nA"]
        self.seed = config.seed
        self.gae_lambda = config.gae_lambda
        self.num_agents = config.num_agents
        self.batch_size = int(config.batch_size * self.num_agents)
        self.tmax = config.tmax
        self.start_epsilon = self.epsilon = config.epsilon
        self.start_beta = self.beta = config.beta
        self.gamma = config.gamma
        self.start_discount = config.discount_rate
        self.gradient_clip = config.gradient_clip
        self.SGD_epoch = config.SGD_epoch
        self.device = "cpu"
        self.network = PPO(config.seed, self.nA)
        count_parameters(self.network)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)

    def forward(self, state):
        action, prob, probs, entropy, value = self.network.forward(state)
        chosen_word = self.dictionary[action.item()]
        return {
            Outputs.ACTION: action,
            Outputs.VALUES: value,
            Outputs.WORD: chosen_word,
            Outputs.ACTION_PROB: prob,
            Outputs.ACTION_PROBS: probs,
        }

    def return_advs(self, values, last_state, rewards):
        N = len(values)
        A = self.num_agents
        self.network.eval()
        # Get values and next_values in the same shape to quickly calculate TD_errors
        with torch.no_grad():
            next_value = self.network(last_state.unsqueeze(0))[-1].cpu()
        self.network.train()
        next_values = torch.cat((values[1:], next_value))
        combined = self.gamma * self.gae_lambda

        next_values = next_values.view(N, A)
        values = values.view(N, A)
        TD_errors = rewards + next_values - values
        advs = np.zeros(rewards.shape)
        returns = np.zeros(rewards.shape)
        rewards = rewards.numpy()
        TD_errors = TD_errors.detach().numpy()

        # For returns
        discounted_gamma = self.gamma ** np.arange(N)
        j = 1
        for index in reversed(range(N)):
            P = N - index
            discounts = combined ** np.arange(0, N - index)
            returns[index, :] = np.sum(
                rewards[index:, :]
                * np.repeat([discounted_gamma[:j]], A, axis=1).reshape(P, A),
                axis=0,
            )
            advs[index, :] = np.sum(
                TD_errors[index:, :] * np.repeat([discounts], A, axis=1).reshape(P, A),
                axis=0,
            )
            j += 1
        # Normalize and reshape
        returns = torch.from_numpy(returns.reshape(N * A, 1)).float().to(self.device)
        advs = torch.from_numpy(advs.reshape(N * A, 1)).float().to(self.device)
        std = 1 if N == 1 else advs.std()
        advs = (advs - advs.mean()) / std
        return advs, rewards, returns

    def minibatch(self, N):
        indicies = np.arange(N - self.batch_size)
        for _ in range(self.SGD_epoch):
            yield np.random.choice(indicies)

    def reset_hyperparams(self):
        self.discount = self.start_discount
        self.epsilon = self.start_epsilon
        self.beta = self.start_beta

    def step_hyperparams(self):
        self.epsilon *= 0.999
        self.beta *= 0.995

    def backward_mcts(
        self,
        actions,
        states,
        rewards,
    ):
        criterion = CrossEntropyLoss()
        _, _, probs, _, values = self.network(states, actions)
        critic_loss = F.mse_loss(values, rewards)
        policy_loss = criterion(probs, actions.long())
        loss = critic_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()

    def backward(
        self,
        actions,
        action_prob,
        action_probs,
        states,
        next_states,
        rewards,
        values,
        dones,
    ):
        advantages, rewards, returns = self.return_advs(
            values, next_states[-1], rewards
        )
        # reshape so that memory is shared
        N = len(states) * self.num_agents
        states = states.to(self.device)
        actions = actions.to(self.device)
        log_probs = action_prob.to(self.device)
        # values = torch.from_numpy((values).reshape(N,1)).float().to(self.device)
        # do multiple training runs on the data
        # for _ in range(self.SGD_epoch):
        #     # Iterate through random batch generator
        #     for start in self.minibatch(N):
        #         # For training on sequences
        #         end = start + self.batch_size
        #         states_b = states[start:end]
        #         actions_b = actions[start:end]
        #         log_probs_b = log_probs[start:end]
        #         returns_b = returns[start:end]
        #         advantages_b = advantages[start:end]
        # for training on random batches (minibatches must be modified)
        # states_b = states[indicies]
        # actions_b = actions[indicies]
        # log_probs_b = log_probs[indicies]
        # values_b = values[indicies]
        # returns_b = returns[indicies]
        # advantages_b = advantages[indicies]

        # get new probabilities with grad to perform the update step
        # Calculate the ratio between old and new log probs. Get loss and update grads
        _, new_log_probs, probs, entropy, new_values = self.network(states, actions)

        ratio = (new_log_probs - log_probs).exp()
        # ratio = new_log_probs / log_probs_b

        clip = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        clipped_surrogate = torch.min(ratio * advantages, clip * advantages)

        actor_loss = -torch.mean(clipped_surrogate) - self.beta * entropy.mean()
        critic_loss = F.smooth_l1_loss(returns, new_values.view(N, -1))

        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.step_hyperparams()

    def learn(self, player_data):
        action_prob = player_data[AgentData.ACTION_PROB]
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
        action_prob = torch.stack(action_prob)
        values = torch.stack(values).squeeze(1)
        self.backward(
            actions,
            action_prob,
            action_probs,
            states,
            next_states,
            rewards,
            values,
            dones,
        )
        # self.backward_mcts(actions, states, rewards)
