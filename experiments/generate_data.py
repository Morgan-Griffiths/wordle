import numpy as np
import copy
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


def stage_n(n):
    if n == 0:
        return np.zeros((6, 5, 2)), 0
    elif n == 5:
        letters = np.random.randint(1, 27, (5, 5, 1))
        results = np.random.randint(1, 4, (5, 5, 1))
        a = np.concatenate([letters, results], -1)
        zeros = np.zeros((1, 5, 2))
        return np.vstack((a, zeros)), 1
    else:
        letters = np.random.randint(1, 27, (n, 5, 1))
        results = np.random.randint(1, 4, (n, 5, 1))
        a = np.concatenate([letters, results], -1)
        zeros = np.zeros((6 - n, 5, 2))
        return np.vstack((a, zeros)), 0


def threshhold_dataset():
    # 6,5,2
    x = []
    y = []
    for _ in range(2500):
        n = np.random.randint(0, 6)
        state, target = stage_n(n)
        x.append(state)
        y.append(target)
    return np.stack(x), np.stack(y)[:, None]


def policy_dataset():
    # 6,5,2
    decay_factor = 0.95
    num_words = 5
    states = []
    targets = []
    rewards = []
    env = Wordle()
    for _ in range(50):
        state, reward, done = env.reset()
        word = np.random.choice(dictionary[:num_words])
        idx = dictionary.index(word)
        env.word = word
        while not done:
            word_idx = np.random.choice(num_words)
            states.append(copy.deepcopy(state))
            state, reward, done = env.step(dictionary[word_idx])
            targets.append(idx)
        temp = [reward] * env.turn
        rewards.extend([r * decay_factor ** i for i, r in enumerate(temp[::-1])][::-1])
    states = np.stack(states)
    targets = np.stack(targets)
    rewards = np.stack(rewards)
    return np.stack(states), (np.stack(targets), np.stack(rewards)[:, None])


def random_dataset():
    print("gathering random data")
    num_words = 5
    states = []
    actions = []
    targets = []
    reward_targets = []
    env = Wordle()
    for _ in range(50):
        state, rewards, done = env.reset()
        env.word = np.random.choice(dictionary[:num_words])
        while not done:
            word_idx = np.random.choice(num_words)
            states.append(copy.deepcopy(state))
            actions.append(word_idx)
            state, rewards, done = env.step(dictionary[word_idx])
            target = state[env.turn - 1, :, 1]
            targets.append(result_index_dict[tuple(target)])
            reward_targets.append(rewards)
    states = np.stack(states)
    actions = np.stack(actions)
    targets = np.stack(targets)
    reward_targets = np.stack(reward_targets)
    return (states, actions), (targets, reward_targets)


def load_data(datatype):
    if datatype == DataTypes.THRESHOLD:
        x, y = threshhold_dataset()
    elif datatype == DataTypes.RANDOM:
        x, y = random_dataset()
    elif datatype == DataTypes.POLICY:
        x, y = policy_dataset()
    else:
        raise ValueError(f"Unknown datatype {datatype}")
    print(x[0].shape, x[1].shape) if isinstance(x, tuple) else print(x.shape)
    print(y[0].shape, y[1].shape) if isinstance(y, tuple) else print(y.shape)
    return {
        "trainX": x,
        "trainY": y,
        "valX": x,
        "valY": y,
    }
