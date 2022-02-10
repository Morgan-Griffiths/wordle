from re import S
import numpy as np
from ML.networks import Letters
from experiments.globals import DataTypes
from globals import (
    State,
    dictionary,
    alphabet,
    alphabet_dict,
    dictionary_word_to_index,
    result_index_dict,
)
from wordle import Wordle
from itertools import permutations


def generate_letter_mapping_data():
    inputs = []
    targets = []
    for i in range(26):
        vector = np.zeros(26)
        vector[i] = 1
        targets.append(i)
        inputs.append(vector)
    return np.vstack(inputs), np.vstack(targets)


def generate_capital_letter_mapping_data():
    inputs = []
    targets = []
    for i in range(26):  # lowercase
        vector = np.zeros(26)
        vector[i] = 1
        targets.append(i)
        inputs.append(vector)
    for i in range(26):  # capitals
        vector = np.zeros(26)
        vector[i] = 2
        targets.append(i + 26)
        inputs.append(vector)
    return np.vstack(inputs), np.vstack(targets)


def generate_wordle_mapping_data():
    states = []
    actions = []
    targets = []
    reward_targets = []
    env = Wordle()
    permut_0 = [permutations(range(0, 5), i) for i in range(6)]
    all_permutations = [permut_0]
    all_targets = list(range(5))
    for permut in all_permutations:
        for target_word in dictionary:
            for perm in permut:
                for subset in perm:
                    state, rewards, done = env.reset()
                    env.word = target_word
                    for word_idx in subset:
                        while not done:
                            state, rewards, done = env.step(dictionary[word_idx])
                            target = state[env.turn - 1, :, 1]
                            states.append(state)
                            actions.append(word_idx)
                            targets.append(result_index_dict[tuple(target)])
                            reward_targets.append(rewards)
    states = np.stack(states)
    actions = np.stack(actions)
    targets = np.stack(targets)
    reward_targets = np.stack(reward_targets)
    return (states, actions), (targets, reward_targets)


def generate_constellation_data():
    # goal is to design a dataset that reflects the decisions required to solve the actual game.
    # the ability to discriminate between
    # 3 words
    # inital distribution should be equal
    # "MOUNT",
    # "HELLO",
    # "NIXED",
    # "AAHED",
    # "HELMS",
    # HELLO -> Nixed or MOUNT
    inputs = []
    targets = []
    env = Wordle()
    for w in ["HELLO", "NIXED"]:
        env.reset()
        env.word = "MOUNT"
        env.step(w)
        inputs.append(env.state)
        targets.append(dictionary_word_to_index[env.word])
    for w in ["HELMS", "MOUNT"]:
        env.reset()
        env.word = "HELLO"
        env.step(w)
        inputs.append(env.state)
        targets.append(dictionary_word_to_index[env.word])
    for w in ["AAHED", "NIXED"]:
        env.reset()
        env.word = "HELMS"
        env.step(w)
        inputs.append(env.state)
        targets.append(dictionary_word_to_index[env.word])
    for w in ["MOUNT", "NIXED"]:
        env.reset()
        env.word = "AAHED"
        env.step(w)
        inputs.append(env.state)
        targets.append(dictionary_word_to_index[env.word])
    for w in [
        "MOUNT",
        "HELLO",
        "NIXED",
        "AAHED",
        "HELMS",
    ]:
        env.reset()
        env.word = w
        inputs.append(env.state)
        targets.append(dictionary_word_to_index[env.word])
    return np.stack(inputs), np.vstack(targets)


def generate_multitarget_wordle_data():
    inputs = []
    targets = []
    env = Wordle()
    permut_0 = [permutations(range(1, 5), i) for i in range(4, 5)]
    permut_1 = [permutations([0, 2, 3, 4], i) for i in range(4, 5)]
    permut_2 = [permutations([0, 1, 3, 4], i) for i in range(4, 5)]
    permut_3 = [permutations([0, 1, 2, 4], i) for i in range(4, 5)]
    permut_4 = [permutations([0, 1, 2, 3], i) for i in range(4, 5)]
    all_permutations = [permut_0, permut_1, permut_2, permut_3, permut_4]
    all_targets = list(range(5))
    for i, permut in enumerate(all_permutations):
        target = all_targets[i]
        for perm in permut:
            for subset in perm:
                env.reset()
                env.word = dictionary[i]
                for word_idx in subset:
                    state, rewards, done = env.step(dictionary[word_idx])
                    inputs.append(state)
                    targets.append(target)
    return np.stack(inputs), np.vstack(targets)


def load_data(datatype):
    if datatype == DataTypes.CAPITALS:
        x, y = generate_capital_letter_mapping_data()
    elif datatype == DataTypes.LETTERS:
        x, y = generate_letter_mapping_data()
    elif datatype == DataTypes.WORDLE:
        x, y = generate_wordle_mapping_data()
    elif datatype == DataTypes.MULTI_TARGET:
        x, y = generate_multitarget_wordle_data()
    elif datatype == DataTypes.CONSTELLATION:
        x, y = generate_constellation_data()
    # print(x.shape, y.shape)
    return {
        "trainX": x,
        "trainY": y,
        "valX": x,
        "valY": y,
    }
