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


def create_root(state, reward):
    root = Node(None, 1)
    root.state = torch.as_tensor(state).long().unsqueeze(0)
    root.reward = reward
    return root


class MCTS:
    def __init__(self, config) -> None:
        self.config = config
        self.epsilon = config.epsilon

    def decay_epsilon(self):
        self.epsilon = max(0, self.epsilon * 0.999)

    def run(self, agent: MuZeroNet, state, reward):
        root = create_root(state, reward)
        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            with torch.no_grad():
                current_tree_depth = 0
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
                        action = np.random.randint(self.config.action_space)
                    else:
                        # network picks
                        action = np.random.choice(
                            len(node.action_probs), p=node.action_probs.numpy()
                        )
                    # if action previous unexplored, expand node
                    if action not in node.children:
                        node.children[action] = Node(
                            parent=node, prior=node.action_probs[action]
                        )
                        outputs: DynamicOutputs = agent.dynamics(
                            node.state,
                            torch.as_tensor(action).view(1, 1),
                        )
                        node.children[action].state_probs = outputs.state_probs.numpy()[
                            0
                        ]
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
                    node.state = torch.as_tensor(next_state).long()
                    current_tree_depth += 1
                max_tree_depth = max(max_tree_depth, current_tree_depth)
                outputs: PolicyOutputs = agent.policy(node.state)
                node.backprop(outputs.value.item(), self.config)
        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root.value,
        }
        return root, extra_info


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.result_history = []
        self.word_history = []
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.child_visits = []
        self.root_values = []
        self.max_actions = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root: Node, action_space: int):
        # Turn visit count from root into a policy
        if root is not None:
            visit_counts = [child.visit_count for child in root.children.values()]
            sum_visits = sum(visit_counts)
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in range(action_space)
                ]
            )

            self.root_values.append(root.value)
            actions = [action for action in root.children.keys()]
            self.max_actions.append(actions[np.argmax(visit_counts)])
        else:
            self.root_values.append(None)

    def prepare_inputs(
        self,
        states,
        actions,
        results,
        reward_targets,
    ):
        states = np.array(states)
        actions = np.array(actions)
        reward_targets = np.array(reward_targets)
        result_targets = [
            torch.as_tensor(result_index_dict[tuple(res)]) for res in results
        ]

        assert (
            len(states) == len(actions) == len(result_targets) == len(reward_targets)
        ), f"Improper lengths {print(len(states), len(actions), len(result_targets), len(reward_targets))}"
        # np.save("states.npy", states)
        # np.save("actions.npy", actions)
        # np.save("reward_targets.npy", reward_targets)
        states = torch.as_tensor(states).long()
        actions = torch.as_tensor(actions).long()
        result_targets = torch.stack(result_targets).long()
        reward_targets = torch.as_tensor(reward_targets).float()
        return states, actions, result_targets, reward_targets
