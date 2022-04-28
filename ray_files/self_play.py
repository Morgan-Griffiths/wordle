import time
from typing import Any
import numpy as np
import copy
from MCTS_mu import MCTS, GameHistory
from ML.agents.mu_agent import MuAgent
from globals import (
    Dims,
    DynamicOutputs,
    Embeddings,
    NetworkOutput,
    PolicyOutputs,
    dictionary_index_to_word,
    dictionary_word_to_index,
    result_index_dict,
    index_result_dict,
)
from utils import to_tensor, state_transition
from ML.networks import MuZeroNet
from wordle import Wordle
import math
import torch
import ray

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


@ray.remote
class SelfPlay:
    def __init__(self, initial_checkpoint, env, config, seed) -> None:
        self.epsilon = 0.5
        self.config = config
        self.env = env

        # Fix random generator seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = MuZeroNet(config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

    def decay_epsilon(self):
        self.epsilon = max(0, self.epsilon * 0.999)

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                )

                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": np.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)

        # self.close_game()

    def play_game(self, temperature, temperature_threshold, render):
        game_history = GameHistory()
        state, reward, done = self.env.reset()
        game_history.state_history.append(state)
        game_history.reward_history.append(reward)
        game_history.word_history.append(self.env.word_to_action(self.env.word))
        with torch.no_grad():
            while not done:
                root, mcts_info = MCTS(self.config).run(
                    self.model,
                    state,
                    reward,
                )
                # get chosen action
                action = self.select_action(
                    root,
                    temperature
                    if not temperature_threshold
                    or len(game_history.action_history) < temperature_threshold
                    else 0,
                )
                if render:
                    print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                    print(f"Root value {root.value:.2f}")
                state, reward, done = self.env.step(dictionary_index_to_word[action])
                if render:
                    print(f"Played action: {self.env.action_to_string(action)}")
                    self.env.visualize_state()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.result_history.append(
                    state[self.env.turn - 1, :, Embeddings.RESULT]
                )
                game_history.word_history.append(
                    dictionary_word_to_index[self.env.word]
                )
                game_history.action_history.append(action)
                game_history.state_history.append(state)
                game_history.reward_history.append(reward)

        return game_history