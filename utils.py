import torch
import copy
import numpy as np
from config import Config
from globals import Dims, Embeddings, Mappings
from prettytable import PrettyTable


def result_from_state(turn, state, mappings: Mappings):
    try:
        result = state[turn][:, Embeddings.RESULT]
        return mappings.result_index_dict[tuple(result)]
    except:
        return -1


def debug(func):
    def wrapper(*args, **kwags):
        print(f"Running {func.__name__}")
        return func(*args, **kwags)

    return wrapper


def state_transition(
    state: np.array, word: str, result: np.array, mappings: Mappings
) -> np.array:
    new_state = copy.deepcopy(state)
    encoded_word = np.array([mappings.alphabet_dict[letter] for letter in word.lower()])
    mask = np.where(new_state == 0)[1]
    turn = min(mask)
    new_state[:, turn, :, Embeddings.LETTER] = encoded_word
    new_state[:, turn, :, Embeddings.RESULT] = result
    return new_state


def load_n_letter_words(n):
    with open("/usr/share/dict/words", "r") as f:
        data = f.read()
    words = data.split("\n")
    cleaned = [word.strip() for word in words if len(word) == n]
    return cleaned


def to_tensor(state):
    return torch.as_tensor(state, dtype=torch.int32)


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
        action_space = range(self.config.action_space)
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
