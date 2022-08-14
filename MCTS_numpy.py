import numpy as np
import math
import torch

from globals import (
    DynamicOutputs,
    Mappings,
    PolicyOutputs,
)
from utils import state_transition
from ML.networks import MuZeroNet
from collections import defaultdict
from memory_profiler import profile


def create_root(state, action_space, reward):
    root = NumpyNode(0, action_space, None, 1)
    root.state = torch.as_tensor(state).long().unsqueeze(0)
    root.reward = reward
    return root


class DummyNode(object):
    """A fake node of a MCTS search tree.
    This node is intended to be a placeholder for the root node, which would
    otherwise have no parent node. If all nodes have parents, code becomes
    simpler."""

    def __init__(self):
        self.parent = None
        self.child_total_value = defaultdict(float)
        self.child_visit_count = defaultdict(float)


class NumpyNode:
    def __init__(self, action, action_space, parent, prior) -> None:
        if parent is None:
            parent = DummyNode()
        self.parent = parent
        self.prior = prior
        self.action = action
        self.action_idx = action - 1
        self.action_probs = None
        self.state_probs = None
        self.reward_outcomes = None
        self.children = {}
        self.state = None
        self.reward = 0
        self.child_priors = np.zeros([action_space], dtype=np.float32)
        self.child_total_value = np.zeros([action_space], dtype=np.float32)
        self.child_visit_count = np.zeros([action_space], dtype=np.float32)

    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def visit_count(self):
        return self.parent.child_visit_count[self.action_idx]

    @visit_count.setter
    def visit_count(self, value):
        self.parent.child_visit_count[self.action_idx] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action_idx]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action_idx] = value

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

    def expand(self, turn: int, action_space: int, action_probs: np.array):
        # Scale number of actinos across the turns. Sample the top 50% of them from network weights, remaining randomly.
        # num_actions = actions_per_turn(turn,action_space)
        indicies = np.arange(action_space)
        actions = indicies + 1
        action_probs = action_probs[indicies]
        policy_sum = sum(action_probs)
        for action, idx in zip(actions, indicies):
            self.children[action] = NumpyNode(
                action, action_space, self, action_probs[idx] / policy_sum
            )

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
            math.log((child.visit_count + config.pb_c_base + 1) / config.pb_c_base)
            + config.pb_c_init
        )
        pb_c *= math.sqrt(child.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = child.reward + config.discount_rate * child.value
        else:
            value_score = 0
        return prior_score + value_score

    def backprop(self, value, discount_rate):
        node = self
        while node.parent.parent:
            node.visit_count += 1
            node.total_value += value
            value = node.reward + discount_rate * value
            node = node.parent
        node.total_value += value
        node.visit_count += 1


class MCTS_numpy:
    def __init__(self, config) -> None:
        self.config = config
        self.mappings = Mappings(config.word_restriction)
        self.epsilon = config.epsilon

    def decay_epsilon(self):
        self.epsilon = max(0, self.epsilon * 0.999)

    # @profile
    def run(self, agent: MuZeroNet, state, initial_reward, turn):
        device = next(agent.parameters()).device
        with torch.no_grad():
            root = create_root(state, self.config.action_space, initial_reward)
            outputs: PolicyOutputs = agent.policy(root.state.to(device))
            root.expand(
                turn,
                self.config.action_space,
                outputs.probs.cpu().numpy()[0],
            )
            max_tree_depth = 0
            for _ in range(self.config.num_simulations):
                current_tree_depth = 0
                node: NumpyNode = root
                reward = initial_reward
                sim_turn = 1
                while reward not in [1, -1]:
                    if node.action_probs is None:
                        outputs: PolicyOutputs = agent.policy(node.state.to(device))
                        node.action_probs = outputs.probs[0]
                    if not node.expanded():
                        node.expand(
                            turn + sim_turn,
                            self.config.action_space,
                            node.action_probs,
                        )
                        # action = np.random.randint(self.config.action_space) + 1
                        # node = node.children[action]
                    if self.config.add_exploration_noise:
                        root.add_exploration_noise(
                            dirichlet_alpha=self.config.root_dirichlet_alpha,
                            exploration_fraction=self.config.root_exploration_fraction,
                        )
                    action, node = node.select_child(self.config)
                    if node.state_probs is None:
                        outputs: DynamicOutputs = agent.dynamics(
                            node.parent.state.to(device),
                            torch.as_tensor(action).view(1, 1).to(device),
                        )
                        node.state_probs = outputs.state_probs.cpu().numpy()[0]
                        node.reward_outcomes = outputs.rewards[0]
                    # sample state after
                    state_choice = np.random.choice(
                        len(node.state_probs), p=node.state_probs
                    )  # zero padding
                    result, reward = (
                        self.mappings.index_result_dict[state_choice],
                        node.reward_outcomes[state_choice],
                    )
                    # get previous state -> new state
                    next_state = state_transition(
                        node.parent.state.cpu().numpy(),
                        self.mappings.dictionary_index_to_word[action],
                        np.array(result),
                        self.mappings,
                    )
                    if state_choice not in node.children:
                        node.children[state_choice] = NumpyNode(
                            action=-1,
                            action_space=self.config.action_space,
                            parent=node,
                            prior=node.state_probs[state_choice],
                        )
                    node = node.children[state_choice]
                    node.reward = int(reward.item())
                    node.state = torch.as_tensor(next_state).long()
                    current_tree_depth += 1
                    sim_turn += 1
                max_tree_depth = max(max_tree_depth, current_tree_depth)
                outputs: PolicyOutputs = agent.policy(node.state)
                node.backprop(outputs.value.item(), self.config.discount_rate)
            extra_info = {
                "max_tree_depth": max_tree_depth,
                "root_predicted_value": root.value,
            }
            # self.decay_epsilon()
        return root, extra_info
