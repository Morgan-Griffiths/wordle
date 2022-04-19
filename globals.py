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


@dataclass
class DynamicOutputs:
    next_state: torch.tensor
    state_probs: torch.tensor
    reward: int
    rewards: np.array


@dataclass
class PolicyOutputs:
    action: int
    probs: torch.tensor
    value: float


with open("data/allowed_words.txt", "r") as f:
    wordle_dictionary = f.readlines()

permutations = []
for a in range(1, 4):
    for b in range(1, 4):
        for c in range(1, 4):
            for d in range(1, 4):
                for e in range(1, 4):
                    permutations.append((a, b, c, d, e))
result_index_dict = {dist: i for i, dist in enumerate(permutations)}
index_result_dict = {i: dist for dist, i in result_index_dict.items()}
dictionary = [word.strip() for word in wordle_dictionary]
dictionary = dictionary[: Dims.OUTPUT]
dictionary_arr = np.vstack([list(word) for word in dictionary])
# dictionary = [
#     "MOUNT",
#     "HELLO",
#     "NIXED",
#     "AAHED",
#     "HELMS",
# ]
dictionary_word_to_index = {word: i for i, word in enumerate(dictionary)}
dictionary_index_to_word = {i: word for i, word in enumerate(dictionary)}
# print(dictionary)
alphabet = "".join("-abcdefghijklmnopqrstuvwxzy".upper().split())
alphabet_dict = {letter: i for i, letter in enumerate(alphabet)}
index_to_letter_dict = {i: letter for i, letter in enumerate(alphabet)}
readable_result_dict = {
    Tokens.UNKNOWN: "UNKNOWN",
    Tokens.MISSING: "MISSING",
    Tokens.CONTAINED: "CONTAINED",
    Tokens.EXACT: "EXACT",
}
