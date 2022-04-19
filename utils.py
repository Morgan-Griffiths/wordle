from typing import NamedTuple
import torch
import copy
import numpy as np
from config import Config
from globals import (
    AgentData,
    Dims,
    Embeddings,
    Outputs,
    Results,
    dictionary_word_to_index,
    result_index_dict,
    alphabet_dict,
    NetworkOutput,
)
from prettytable import PrettyTable
from scipy.stats import entropy

def state_transition(state:np.array,word:str,result:np.array) -> np.array:
    new_state = copy.deepcopy(state)
    encoded_word = np.array([alphabet_dict[letter] for letter in word.upper()])
    mask = np.where(new_state == 0)[1]
    turn = min(mask)
    new_state[:,turn,:,Embeddings.LETTER] = encoded_word
    new_state[:,turn,:,Embeddings.RESULT] = result
    return new_state

def load_n_letter_words(n):
    with open("/usr/share/dict/words", "r") as f:
        data = f.read()
    words = data.split('\n')
    cleaned = [word.strip() for word in words if len(word) == n]
    return cleaned

def return_rewards(turn: int, reward: float):
    sign = -1 if reward < 0 else 1
    return list(reversed([torch.Tensor([0.95 ** num * sign]) for num in range(turn)]))


def to_tensor(state):
    return torch.as_tensor(state, dtype=torch.int32)


def return_result_params() -> dict:
    return {
        Results.LOSSES: [],
        Results.VALUES: [],
        Results.ACTION_PROBS: [],
        Results.ACTIONS: [],
        Results.TARGETS: [],
    }


def return_data_params():
    return {
        AgentData.STATES: [],
        AgentData.ACTIONS: [],
        AgentData.ACTION_PROB: [],
        AgentData.ACTION_PROBS: [],
        AgentData.VALUES: [],
        AgentData.REWARDS: [],
        AgentData.DONES: [],
        AgentData.TARGETS: [],
    }


def return_permutations():  # 3^5 = 243
    results = []
    for a in range(1, 4):
        for b in range(1, 4):
            for c in range(1, 4):
                for d in range(1, 4):
                    for e in range(1, 4):
                        results.append((a, b, c, d, e))
    return results


def store_state(
    data_params: dict,
    state: np.array,
    done: bool,
    target: str,
) -> dict:
    data_params[AgentData.STATES].append(to_tensor(state))
    data_params[AgentData.DONES].append(to_tensor(done))
    data_params[AgentData.TARGETS].append(to_tensor(dictionary_word_to_index[target]))
    return data_params


def store_outputs(data_params: dict, outputs: dict) -> dict:
    data_params[AgentData.ACTION_PROBS].append(outputs[Outputs.ACTION_PROBS])
    data_params[AgentData.ACTION_PROB].append(outputs[Outputs.ACTION_PROB])
    data_params[AgentData.VALUES].append(outputs[Outputs.VALUES])
    data_params[AgentData.ACTIONS].append(outputs[Outputs.ACTION])
    return data_params


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def shape_values_to_q_values(values, actions):
    data = np.zeros((len(values), Dims.OUTPUT))
    i = 0
    for value, action in zip(values, actions):
        data[i, action] = value
        i += 1
    return data


def select_action(node, temperature=1, deterministic=True):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    action_probs = [
        visit_count_i ** (1 / temperature) for visit_count_i, _ in visit_counts
    ]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        action_pos = np.argmax([v for v, _ in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return visit_counts[action_pos][1], count_entropy


class DynamicsStorage:
    def __init__(self):
        # state
        self.rewards = []
        self.result_targets = []
        # network
        self.projected_rewards = []
        self.projected_results = []

    def store_result_target(self, state, turn):
        if turn > 0:
            target = state[turn - 1, :, 1].astype(int)
            self.result_targets.append(to_tensor(result_index_dict[tuple(target)]))

    def store_rewards(self, rewards):
        self.rewards = rewards

    def store_outputs(self, projected_rewards, projected_results):
        self.projected_rewards.append(projected_rewards)
        self.projected_results.append(projected_results)


class DataStorage:
    def __init__(self):
        # state
        self.states = []
        self.dones = []
        self.rewards = []
        self.result_targets = []
        self.word_targets = []
        # network
        self.actions = []
        self.action_targets = []
        self.policy_logits = []
        self.values = []
        self.projected_reward = []
        self.projected_rewards = []
        self.projected_results = []

    def store_state(self, state, done, word, turn):
        self.states.append(to_tensor(state))
        self.dones.append(to_tensor(done))
        self.word_targets.append(to_tensor(dictionary_word_to_index[word]))
        if turn > 0:
            target = state[turn - 1, :, 1].astype(int)
            self.result_targets.append(to_tensor(result_index_dict[tuple(target)]))

    def store_outputs(self, network_outputs: NetworkOutput, target_action):
        self.policy_logits.append(network_outputs.policy_logits)
        self.actions.append(to_tensor(network_outputs.action))
        self.values.append(network_outputs.value)
        self.projected_reward.append(network_outputs.reward)
        self.projected_rewards.append(network_outputs.rewards)
        self.projected_results.append(network_outputs.result_logits)
        self.action_targets.append(to_tensor(target_action).unsqueeze(0))

    def store_rewards(self, rewards):
        self.rewards.extend(rewards)


class Stats:
    def __init__(self, config: Config) -> None:

        self.config = config
        self.states = []
        self.actions = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []

    def make_target(
        self,
        state_index: int,
        num_unroll_steps: int,
        td_steps: int,
        model=None,
        config=None,
    ):
        # The value target is the discounted root value of the search tree N steps into the future, plus
        # the discounted sum of all rewards until then.
        target_values, target_rewards, target_policies = [], [], []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                if model is None:
                    value = (
                        self.root_values[bootstrap_index]
                        * self.config.discount_rate ** td_steps
                    )
                else:
                    # Reference : Appendix H => Reanalyze
                    # Note : a target network  based on recent parameters is used to provide a fresher,
                    # stable n-step bootstrapped target for the value function
                    state = self.return_states(bootstrap_index)
                    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    network_output = model(state)
                    value = (
                        network_output.value.data.cpu().item()
                        * self.config.discount_rate ** td_steps
                    )
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.config.discount_rate ** i

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = torch.tensor([0])

            if current_index < len(self.root_values):
                target_values.append(value)
                target_rewards.append(last_reward)

                # Reference : Appendix H => Reanalyze
                # Note : MuZero Reanalyze revisits its past time-steps and re-executes its search using the
                # latest model parameters, potentially resulting in a better quality policy than the original search.
                # This fresh policy is used as the policy target for 80% of updates during MuZero training
                if (
                    model is not None
                    and np.random.random() <= self.config.revisit_policy_search_rate
                ):
                    from MCTS_mu import MCTS, Node

                    root = Node(0)
                    state = self.return_states(current_index)
                    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    network_output = model(state)
                    root.expand(Dims.OUTPUT, network_output)
                    MCTS(config).run(root, [current_index], model)
                    self.store_search_stats(root, current_index)

                target_policies.append(self.child_visits[current_index])

            else:
                ...
                # States past the end of games are treated as absorbing states.
                # target_values.append(torch.tensor([0]))
                # target_rewards.append(last_reward)
                # # Note: Target policy is  set to 0 so that no policy loss is calculated for them
                # target_policies.append([0 for _ in range(len(self.child_visits[0]))])

        return (
            torch.stack(target_values),
            torch.stack(target_rewards),
            np.stack(target_policies),
        )

    def store_search_stats(self, root, idx: int = None):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = range(Dims.OUTPUT)
        if idx is None:
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )
            self.root_values.append(root.value)
        else:
            self.child_visits[idx] = [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
            self.root_values[idx] = root.value

    def return_states(self, index):
        return self.states[index:]

    def store_state(self, state):
        self.states.append(to_tensor(state))

    def store_action(self, action):
        self.actions.append(to_tensor(action))

    def store_rewards(self, rewards):
        self.rewards.extend(rewards)

