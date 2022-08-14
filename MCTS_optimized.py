import numpy as np
import math
import torch
from globals import (
    DynamicOutputs,
    Mappings,
    PolicyOutputs,
)

from utils import state_transition, result_from_state
from ML.networks import MuZeroNet
from collections import defaultdict
from memory_profiler import profile


class MCTS_dict:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.epsilon = config.epsilon
        self.mappings = Mappings(config.word_restriction)
        self.Nsa = defaultdict(
            lambda: 0
        )  # stores #times edge s,a was visited. R * 6 * 64 bits
        self.Ns = defaultdict(
            lambda: 0
        )  # stores #times board s was visited. R * 5 * 64 bits
        self.Pa = {}  # stores action probs. R * 5 * 32 bits * 2300
        self.Ps = {}  # stores state probs. R * 6 * 32 bits * 243
        self.Rs = {}  # stores reward outcomes. R * 6 * 243 * 8 bits. values of (-1,0,1)
        self.Vs = defaultdict(lambda: 0)  # stores values. R * 6 * 64 bits

    def select_action(self, s, turn, reward, probs):
        max_ucb = 0
        action = -1
        for a in range(0, self.config.action_space):
            ucb = self.ucb_score(s, a, reward, probs)
            if ucb > max_ucb:
                max_ucb = ucb
                action = a
        if max_ucb == 0:
            action = np.random.randint(0, self.config.action_space)
        return action

    # def select_action(self, s, turn, reward):
    #     ucbs = np.array(
    #         [self.ucb_score(s, a, reward) for a in range(self.config.action_space)]
    #     )
    #     probs = ucbs / np.sum(ucbs)
    #     return np.random.choice(np.arange(0, self.config.action_space), p=probs)

    def ucb_score(self, s, a, reward, probs) -> float:
        s_a = s + str(a)
        pb_c = (
            math.log((self.Ns[s] + self.config.pb_c_base + 1) / self.config.pb_c_base)
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(self.Ns[s]) / (self.Nsa[s_a] + 1)

        prior_score = pb_c * probs[a]
        if self.Nsa[s_a] > 0:
            value_score = reward + self.config.discount_rate * self.Vs[s]
        else:
            value_score = 0
        return prior_score + value_score

    def run(self, agent: MuZeroNet, initial_state, initial_reward, turn):
        device = next(agent.parameters()).device
        with torch.no_grad():
            max_tree_depth = 0
            for _ in range(self.config.num_simulations):
                current_tree_depth = 0
                reward = initial_reward
                sim_turn = turn
                s_key = str(result_from_state(sim_turn, initial_state))
                state_path = [s_key]
                state = torch.as_tensor(initial_state).unsqueeze(0).long()
                while reward not in [1, -1]:
                    self.Ns[s_key] += 1
                    outputs: PolicyOutputs = agent.policy(state.to(device))
                    # self.Pa[s_key] = outputs.probs[0]
                    action_idx = self.select_action(
                        s_key, sim_turn, reward, outputs.probs[0]
                    )
                    action = action_idx + 1
                    state_path.append(str(action))
                    s_a = "".join(state_path)
                    self.Nsa[s_a] += 1
                    if s_a not in self.Ps:
                        outputs: DynamicOutputs = agent.dynamics(
                            state.to(device),
                            torch.as_tensor(action).view(1, 1).to(device),
                        )
                        self.Ps[s_a] = outputs.state_probs.cpu().numpy()[0]
                        self.Rs[s_a] = outputs.rewards[0]

                    state_choice = np.random.choice(
                        len(self.Ps[s_a]), p=self.Ps[s_a]
                    )  # zero padding
                    result, reward = (
                        self.mappings.index_result_dict[state_choice],
                        self.Rs[s_a][state_choice],
                    )
                    # get previous state -> new state
                    next_state = state_transition(
                        state.cpu().numpy(),
                        self.mappings.dictionary_index_to_word[action],
                        np.array(result),
                        self.mappings,
                    )
                    reward = int(reward.item())
                    state = torch.as_tensor(next_state).long()
                    current_tree_depth += 1
                    sim_turn += 1
                    state_path.append(str(state_choice))
                    s_key = "".join(state_path)
                self.Ns[s_key] += 1
                max_tree_depth = max(max_tree_depth, current_tree_depth)
                outputs: PolicyOutputs = agent.policy(state)
                self.backprop(outputs.value.item(), state_path, reward)
            extra_info = {
                "max_tree_depth": max_tree_depth,
                "root_predicted_value": self.Vs[state_path[0]] / self.Ns[state_path[0]],
            }
            # self.decay_epsilon()
        return extra_info

    def backprop(self, value, state_path, reward):
        rewards = [0] * (len(state_path) // 2)
        rewards[-1] = reward
        while len(state_path) > 1:
            s = "".join(state_path)
            state_path.pop()
            r = rewards.pop()
            value = r + self.config.discount_rate * value
            self.Vs[s] += value
            state_path.pop()
        s = "".join(state_path)
        self.Vs[s] += value
