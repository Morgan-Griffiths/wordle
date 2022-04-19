from typing import Any
import numpy as np
import copy
from ML.agents.mu_agent import MuAgent
from globals import (
    Dims,
    DynamicOutputs,
    NetworkOutput,
    PolicyOutputs,
    result_index_dict,
    index_result_dict,
    dictionary_index_to_word,
)
from utils import to_tensor, state_transition
from ML.networks import MuZeroNet
from wordle import Wordle
import math
import torch

""" 
MuZero MCTS:
the initial state is passed through the encoder -> h'. 
To get S', we query the dynamics function of MuZero which takes (S,A) -> P(S').
Because we output a probability, we may have many S' nodes. Each time we query that
part of the tree, we sample from P(S').

Sample action -> sample state, see if state exists, else new node. 
When reward is 1 or -1, propogate 

Action selection
a_t = argmax (Q(s,a) + U(s,a)) 
C: exploration rate
U: C(s) * P(s,a) * sqrt(N(s)) / 1 + N(s,a)
N(s) = parent visit count
C(s) = log((1 + N(s) + C_base)/C_base) + c_init
W(s) = total action value
"""


class Node:
    def __init__(self, parent, prior) -> None:
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.action_probs = None
        self.state_probs = None
        self.reward_outcomes = None
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions: int, network_outputs: NetworkOutput):
        self.state = network_outputs.state
        self.reward = network_outputs.reward
        policy = {
            a: math.exp(network_outputs.policy_logits[0][a]) for a in range(actions)
        }
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            self.children[action] = Node(p / policy_sum)

    def expand_state(self, states: int, network_outputs: NetworkOutput):
        self.reward = network_outputs.reward
        policy = {
            s: math.exp(network_outputs.result_logits[0][s]) for s in range(states)
        }
        policy_sum = sum(policy.values())
        for state, p in policy.items():
            self.children[state] = Node(p / policy_sum)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

    def select_child(self, config):
        # print("lenght", len(self.children))
        # for action, child in self.children.items():
        #     print("check", action, child)
        _, action, child = max(
            (self.ucb_score(child, config), action, child)
            for action, child in self.children.items()
        )
        return action, child

    def ucb_score(self, child, config) -> float:
        pb_c = (
            math.log(
                (child.parent.visit_count + config.pb_c_base + 1) / config.pb_c_base
            )
            + config.pb_c_init
        )
        pb_c *= math.sqrt(child.parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = child.reward + config.discount_rate * child.value
        else:
            value_score = 0
        return prior_score + value_score

    def backprop(self, value, config):
        # print("backprop", value)
        node = self
        while node.parent:
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + config.discount_rate * value
            # print("node attrs", node.value, node.reward)
            # print("value_sum", node.value_sum)
            node = node.parent
        node.value_sum += value
        node.visit_count += 1


class MCTS:
    def __init__(self, config) -> None:
        self.epsilon = 1
        self.config = config

    def decay_epsilon(self):
        self.epsilon = max(0, self.epsilon * 0.999)

    def run(self, root: Node, agent: MuZeroNet):
        for _ in range(self.config.num_simulations):
            node: Node = root
            while node.reward not in [1, -1]:
                if node.action_probs is None:
                    outputs: PolicyOutputs = agent.policy(node.state)
                    node.action_probs = outputs.probs[0]
                # pick action
                # print(len(node.children), node.expanded())
                # if len(node.children) == self.config.ubc_start:
                #     # ucb pick
                #     action, node = node.select_child(self.config)
                # else:
                if np.random.random() < self.epsilon:
                    # random action
                    action = np.random.randint(5)
                else:
                    # network picks
                    action = np.random.choice(
                        len(node.state_probs), p=node.action_probs
                    )
                # action = torch.as_tensor(action).unsqueeze(0)
                # print(dictionary_index_to_word[action.item()])
                # if action previous unexplored, expand node
                if action not in node.children:
                    node.children[action] = Node(
                        parent=node, prior=node.action_probs[action]
                    )
                    outputs: DynamicOutputs = agent.dynamics(
                        node.state, torch.as_tensor(action).unsqueeze(0)
                    )
                    node.children[
                        action
                    ].state_probs = outputs.state_probs.detach().numpy()[0]
                    node.children[action].reward_outcomes = outputs.rewards[0]
                node: Node = node.children[action]
                # sample state after
                state_choice = np.random.choice(
                    len(node.state_probs), p=node.state_probs
                )
                result, reward = (
                    index_result_dict[state_choice],
                    node.reward_outcomes[state_choice],
                )
                # print("result", result)
                # print("reward", reward)
                # print("word", dictionary_index_to_word[action.item()])
                # get previous state -> new state
                next_state = state_transition(
                    node.parent.state.numpy(),
                    dictionary_index_to_word[action],
                    np.array(result),
                )
                # print("next_state", next_state)
                node.children[state_choice] = Node(
                    parent=node, prior=node.state_probs[state_choice]
                )
                node = node.children[state_choice]
                node.reward = int(reward.item())
                node.state = torch.as_tensor(next_state)
            outputs: PolicyOutputs = agent.policy(to_tensor(node.state))
            # print(outputs)
            node.backprop(outputs.value.item(), self.config)
