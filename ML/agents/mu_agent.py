import os
from ML.networks import MuZeroNet, StateActionTransition, StateEncoder, ZeroPolicy
from ML.utils import hard_update
from config import Config
import torch
from globals import (
    Dims,
    Embeddings,
    dictionary,
    dictionary_index_to_word,
    alphabet_dict,
    NetworkOutput,
    result_index_dict,
    index_result_dict,
)
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from utils import DataStorage, to_tensor


def update_state(result, old_state, action):
    state = old_state.clone().squeeze(0)
    turn = np.min(np.where(state[:, 0, 0] == 0)[0])
    letters = [alphabet_dict[l] for l in dictionary_index_to_word[action.item()]]
    state[turn, :, Embeddings.RESULT] = to_tensor(result).long()
    state[turn, :, Embeddings.LETTER] = to_tensor(letters).long()
    return state.unsqueeze(0)


class MuAgent:
    def __init__(self, params, config: Config):
        self.dictionary = {i: word.strip() for i, word in enumerate(dictionary)}
        self.nA = Dims.OUTPUT
        self.gradient_clip = config.gradient_clip
        self.seed = config.seed
        self.config = config
        self.criterion = CrossEntropyLoss()
        self.network = MuZeroNet(config, params, output_dims=Dims.OUTPUT)
        self.target_network = MuZeroNet(config, params, output_dims=Dims.OUTPUT)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        # Copy the weights from local to target
        hard_update(self.network, self.target_network)

    def __call__(self, state):
        if isinstance(state, np.ndarray):
            state = to_tensor(state).unsqueeze(0)
        selected_action, action_probs, value = self.network.policy(state)
        return NetworkOutput(
            value,
            np.zeros(5),
            0,
            np.zeros(Dims.RESULT_STATE),
            action_probs,
            state,
            selected_action,
            None,
        )

    def target(self, state):
        if isinstance(state, np.ndarray):
            state = to_tensor(state).unsqueeze(0)
        selected_action, action_probs, value = self.target_network.policy(state)
        return NetworkOutput(
            value,
            0,
            np.zeros(5),
            np.zeros(Dims.RESULT_STATE),
            action_probs,
            state,
            selected_action,
            None,
        )

    def state_transition(self, state, action):
        result, result_logits, reward, reward_logits = self.network.dynamics(
            state, action
        )
        state_prime = update_state(result, state, action)
        return state_prime, result, result_logits, reward, reward_logits

    def query_policy(self, state):
        return self.network.policy(state)

    def recurrent_inference(self, state, action):
        state_prime, result, result_logits, reward, rewards = self.state_transition(
            state, action
        )
        selected_action, action_probs, value = self.query_policy(state_prime)
        return NetworkOutput(
            value,
            result,
            reward,
            rewards,
            action_probs,
            state_prime,
            selected_action,
            result_logits,
        )

    def initial_inference(self, state):
        hidden_state = self.representation(state)
        selected_action, action_probs, prob, value = self.query_policy(hidden_state)
        return NetworkOutput(
            value,
            0,
            np.zeros(5),
            np.zeros(Dims.RESULT_STATE),
            action_probs,
            hidden_state,
            selected_action,
            None,
        )

    def learn_dynamics(self, data):
        projected_rewards = torch.stack(data.projected_rewards).squeeze(1)
        rewards = torch.stack(data.rewards).float()
        projected_results = torch.vstack(data.projected_results)
        result_targets = torch.stack(data.result_targets)

        result_loss = self.criterion(projected_results, result_targets.long())
        selected_rewards = projected_rewards.gather(
            1, result_targets.unsqueeze(1).long()
        )
        reward_loss = F.smooth_l1_loss(selected_rewards, rewards)
        loss = result_loss + reward_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()
        return loss.item()

    def backward(
        self,
        actions,
        action_targets,
        policy_logits,
        states,
        next_states,
        values,
        rewards,
        dones,
        result_targets,
        projected_reward,
        projected_rewards,
        projected_results,
        word_targets,
        target_values,
        target_rewards,
        target_policies,
    ):
        value_loss = F.mse_loss(values, target_values)
        policy_loss = -(
            torch.log_softmax(policy_logits, dim=1) * torch.tensor(target_policies)
        ).sum()
        # policy_loss = self.criterion(policy_logits, word_targets.long())
        result_loss = self.criterion(
            projected_results, torch.as_tensor(result_targets).long()
        )
        selected_rewards = projected_rewards.gather(1, actions).squeeze(1)
        reward_loss = F.smooth_l1_loss(
            selected_rewards, torch.as_tensor(rewards).float()
        )
        loss = result_loss + reward_loss + policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.soft_update(self.network, self.target_network, self.config.tau)

    def learn(
        self, player_data: DataStorage, target_values, target_rewards, target_policies
    ):
        policy_logits = player_data.policy_logits
        values = player_data.values
        rewards = player_data.rewards
        actions = player_data.actions
        projected_results = player_data.projected_results
        projected_reward = player_data.projected_reward
        projected_rewards = player_data.projected_rewards
        action_targets = player_data.action_targets
        states = player_data.states
        dones = player_data.dones
        word_targets = player_data.word_targets
        result_targets = player_data.result_targets
        dones = torch.stack(dones)
        rewards = torch.stack(rewards)
        word_targets = torch.stack(word_targets[1:])
        projected_reward = torch.stack(projected_reward).squeeze(-1)
        projected_rewards = torch.stack(projected_rewards).squeeze(-1)
        action_targets = torch.stack(action_targets).squeeze(-1)
        projected_results = torch.stack(projected_results).squeeze(1)
        result_targets = torch.stack(result_targets)
        next_states = torch.stack(states[1:])
        states = torch.stack(states[:-1])
        actions = torch.stack(actions)
        policy_logits = torch.stack(policy_logits).squeeze(1)
        values = torch.stack(values).squeeze(1)
        self.backward(
            actions,
            action_targets,
            policy_logits,
            states,
            next_states,
            values,
            rewards,
            dones,
            result_targets,
            projected_reward,
            projected_rewards,
            projected_results,
            word_targets,
            target_values,
            target_rewards,
            target_policies,
        )

    def load_weights(self, path):
        self.network.load_state_dict(torch.load(path))

    def save_weights(self, path):
        # print(f"saving weights to {path}")
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.network.state_dict(), path)

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
