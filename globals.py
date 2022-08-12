from dataclasses import dataclass
from typing import Any, NamedTuple
import numpy as np
import torch


class Tokens:
    UNKNOWN = 0
    MISSING = 1
    CONTAINED = 2
    EXACT = 3


class Results:
    VALUES = "values"
    ACTION_PROBS = "action_probs"
    ACTIONS = "actions"
    LOSSES = "losses"
    TARGETS = "targets"


class Axis:
    SLOT = 0
    STATE = 1
    LETTER = 2
    TURN = 3


class Models:
    Q_LEARNING = "q_learning"
    AC_LEARNING = "ac_learning"
    REINFORCE = "reinforce"
    PPO = "ppo"
    POLICY = "policy"
    MUZERO = "muzero"


class Outputs:
    VALUES = "values"
    ACTION_PROBS = "action_probs"
    ACTION_PROB = "action_prob"
    ACTION = "action"
    WORD = "word"


class AgentData:
    REWARDS = "rewards"
    ACTIONS = "actions"
    ACTION_PROBS = "action_probs"
    ACTION_PROB = "action_prob"
    VALUES = "values"
    STATES = "states"
    DONES = "dones"
    TARGETS = "targets"


class Embeddings:
    LETTER = 0
    RESULT = 1
    WORD = 2


class State:
    SHAPE = (6, 5, 2)


class Dims:
    INPUT = 26
    OUTPUT = 500
    HIDDEN_STATE = 128
    RESULT_STATE = 243
    EMBEDDING_SIZE = 8
    TRANSFORMER_INPUT = 40  # 5 * EMBEDDING_SIZE
    TRANSFORMER_OUTPUT = 500
    PROCESSED = EMBEDDING_SIZE * 31


class Train:
    MCTS = "mcts"
    REGULAR = "regular"
    DYNAMICS = "dynamics"


class NetworkOutput(NamedTuple):
    value: float
    result: Any
    reward: float
    rewards: Any
    policy_logits: Any
    state: Any
    action: int
    result_logits: Any


class DynamicOutputs:
    def __init__(
        self, state_logprobs: torch.Tensor, state_probs: torch.Tensor, rewards: np.array
    ):
        self.state_logprobs = state_logprobs
        self.state_probs = state_probs
        self.rewards = rewards
        self.n = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < 2:
            return (self.state_logprobs, self.state_probs, self.rewards)[self.n]
        else:
            raise StopIteration


class PolicyOutputs:
    def __init__(self, action, logprobs, probs, value):
        self.action = action
        self.logprobs = logprobs
        self.probs = probs
        self.value = value
        self.n = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < 3:
            return (self.action, self.logprobs, self.probs, self.value)[self.n]
        else:
            raise StopIteration


CHECKPOINT = {
    "weights": None,
    "optimizer_state": None,
    "total_reward": 0,
    "muzero_reward": 0,
    "opponent_reward": 0,
    "episode_length": 0,
    "mean_value": 0,
    "training_step": 0,
    "lr": 0,
    "total_loss": 0,
    "actor_loss": 0,
    "policy_loss": 0,
    "value_loss": 0,
    "dynamic_loss": 0,
    "actor_probs": 0,
    "actor_value": 0,
    "dynamic_prob_winning_state": 0,
    "actions": 0,
    "results": 0,
    "num_played_games": 0,
    "num_played_steps": 0,
    "num_reanalysed_games": 0,
    "terminate": False,
}

alphabet = "".join("-abcdefghijklmnopqrstuvwxzy".lower().split())


class Mappings:
    """Class to hold wordle result and word mappings"""

    def __init__(self, word_restriction=None):
        # RESULTS
        permutations = []  # zero padded
        for a in range(1, 4):
            for b in range(1, 4):
                for c in range(1, 4):
                    for d in range(1, 4):
                        for e in range(1, 4):
                            permutations.append((a, b, c, d, e))
        self.result_index_dict = {dist: i for i, dist in enumerate(permutations)}
        self.index_result_dict = {i: dist for dist, i in self.result_index_dict.items()}

        # WORDS
        with open("data/allowed_words.txt", "r") as f:
            wordle_dictionary = f.readlines()
        self.dictionary = [word.strip() for word in wordle_dictionary]

        with open("data/possible_words.txt", "r") as f:
            word_targets_dictionary = f.readlines()
        self.target_dictionary = [word.strip() for word in word_targets_dictionary]

        if word_restriction is not None:
            step_size = len(self.dictionary) // word_restriction
            self.dictionary_in_use = self.dictionary[::step_size][:word_restriction]
        else:
            self.dictionary_in_use = self.target_dictionary

        self.dictionary_word_to_index = {
            word: i for i, word in enumerate(self.dictionary_in_use, 1)
        }
        self.dictionary_index_to_word = {
            i: word for i, word in enumerate(self.dictionary_in_use, 1)
        }
        self.dictionary_index_to_word[0] = "-----"
        self.dictionary_word_to_index["-----"] = 0
        self.readable_result_dict = {
            Tokens.UNKNOWN: "UNKNOWN",
            Tokens.MISSING: "MISSING",
            Tokens.CONTAINED: "CONTAINED",
            Tokens.EXACT: "EXACT",
        }

        # LETTERS
        self.alphabet_dict = {letter: i for i, letter in enumerate(alphabet)}
        self.index_to_letter_dict = {
            i: letter for i, letter in enumerate(alphabet)
        }

    def action_to_string(self, action: int):
        try:
            return self.dictionary_index_to_word[action]
        except:
            raise ValueError(f"Invalid action {action}")

    def word_to_action(self, word: str):
        try:
            return self.dictionary_word_to_index[word.lower()]
        except:
            raise ValueError(f"Invalid word {word}")
