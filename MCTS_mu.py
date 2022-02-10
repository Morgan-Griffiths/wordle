from dataclasses import dataclass
from queue import PriorityQueue
from typing import Any
import numpy as np
from copy import copy
from pyrsistent import v
from ML.agents.base_agent import Agent
from ML.agents.mu_agent import MuAgent
from globals import Dims, NetworkOutput, Outputs, result_index_dict
from utils import to_tensor
from wordle import Wordle
import math

""" 
Action selection
a_t = argmax (Q(s,a) + U(s,a)) 
C: exploration rate
U: C(s) * P(s,a) * sqrt(N(s)) / 1 + N(s,a)
N(s) = parent visit count
C(s) = log((1 + N(s) + C_base)/C_base) + c_init
W(s) = total action value
"""


class Node:
    def __init__(self, prior: float) -> None:
        self.visit_count = 0
        self.prior = prior
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


class MCTS:
    def __init__(self, config) -> None:
        self.config = config

    def run(self, root: Node, action_history: list, agent: MuAgent):
        for _ in range(self.config.num_simulations):
            history = []
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                history.append(action)
                search_path.append(node)

            parent = search_path[-2]
            if not len(
                np.where(parent.state[:, :, 0, 0] == 0)[0]
            ):  # break when reach end of game
                break
            network_output = agent.recurrent_inference(
                parent.state, to_tensor(history[-1]).unsqueeze(0)
            )
            # Expand state distribution
            node.expand_state(Dims.RESULT_STATE, network_output)
            # pick state
            index = result_index_dict[network_output.result]

            node.expand_state(Dims.RESULT_STATE, network_output)

            self.backprop(search_path, network_output.value.item())
            action_history.append(history)
        return history

    def select_child(self, node):
        _, action, child = max(
            (self.ucb_score(node, child), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def ucb_score(self, parent, child) -> float:
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = child.reward + self.config.discount_rate * child.value
        else:
            value_score = 0
        return prior_score + value_score

    def backprop(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.config.discount_rate * value
