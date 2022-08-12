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
from memory_profiler import profile

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


def top_k_actions(probs, k):
    highest_indicies = np.argpartition(probs[None, :], -k)[0][-k:]
    all_indicies = set(range(len(probs)))
    remaining = np.array(all_indicies - set(highest_indicies))
    additional_indicies = np.random.choice(remaining, k)
    combined = np.concatenate([highest_indicies, additional_indicies])
    freqs = probs[combined]


""" i have a few options. i expand over all the nodes like before, but renormalize the chosen nodes and change the rest to 0. 
Or i only expand up to the desired number, and then on the learning pass, i pad out the action tree in creating the policy targets"""


def create_root(state, reward):
    root = Node(None, 1)
    root.state = torch.as_tensor(state).long().unsqueeze(0)
    root.reward = reward
    return root


def actions_per_turn(turn, action_space):
    if turn > 1:
        return action_space
    elif turn == 0:
        return 10
    elif turn == 1:
        return min(action_space, 100)


def select_state(outputs):
    state_probs = outputs.state_probs.cpu().numpy()[0]
    reward_outcomes = outputs.rewards[0]
    # sample state after
    state_choice = np.random.choice(len(state_probs), p=state_probs)  # zero padding
    result, reward = (
        i[state_choice],
        reward_outcomes[state_choice],
    )
    return result, reward


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

    def expand(self, turn: int, action_space: int, action_probs: np.array):
        # num_actions = actions_per_turn(turn,action_space)
        indicies = np.arange(action_space)
        actions = indicies + 1
        action_probs = action_probs[indicies]
        policy_sum = sum(action_probs)
        for action, idx in zip(actions, indicies):
            self.children[action] = Node(self, action_probs[idx] / policy_sum)

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

    def backprop(self, value, discount_rate):
        # print("backprop", value)
        node = self
        while node.parent:
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + discount_rate * value
            # print("node attrs", node.value, node.reward)
            # print("value_sum", node.value_sum)
            node = node.parent
        node.value_sum += value
        node.visit_count += 1


class MCTS:
    def __init__(self, config) -> None:
        self.config = config
        self.epsilon = config.epsilon
        self.mappings = Mappings(config.word_restriction)

    def decay_epsilon(self):
        self.epsilon = max(0, self.epsilon * 0.999)

    @profile
    def run(self, agent: MuZeroNet, state, initial_reward, turn):
        device = next(agent.parameters()).device
        with torch.no_grad():
            root = create_root(state, initial_reward)
            outputs: PolicyOutputs = agent.policy(root.state.to(device))
            root.expand(turn, self.config.action_space, outputs.probs.cpu().numpy()[0])
            max_tree_depth = 0
            for _ in range(self.config.num_simulations):
                current_tree_depth = 0
                node: Node = root
                reward = initial_reward
                sim_turn = 1
                while reward not in [1, -1]:
                    if node.action_probs is None:
                        outputs: PolicyOutputs = agent.policy(node.state.to(device))
                        node.action_probs = outputs.probs[0]
                    if not node.expanded():
                        node.expand(
                            turn + sim_turn, self.config.action_space, node.action_probs
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
                        self.config.index_to_word[action],
                        np.array(result),
                    )
                    if state_choice not in node.children:
                        node.children[state_choice] = Node(
                            parent=node, prior=node.state_probs[state_choice]
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


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self, mappings):
        self.mappings = mappings
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
                    for a in range(1, action_space + 1)
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
            torch.as_tensor(self.mappings.result_index_dict[tuple(res)])
            for res in results
        ]

        assert (
            len(states) == len(actions) == len(result_targets) == len(reward_targets)
        ), f"Improper lengths {print(len(states), len(actions), len(result_targets), len(reward_targets))}"
        # np.save("data/states.npy", states)
        # np.save("data/actions.npy", actions)
        # np.save("data/reward_targets.npy", reward_targets)
        states = torch.as_tensor(states).long()
        actions = torch.as_tensor(actions).long()
        result_targets = torch.stack(result_targets).long()
        reward_targets = torch.as_tensor(reward_targets).float()
        return states, actions, result_targets, reward_targets
