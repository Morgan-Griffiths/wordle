from ML.utils import reward_over_states
from globals import WordDictionaries
from utils import state_transition
import torch
import numpy as np


def test_state_transition(word_dictionary: WordDictionaries):
    state = np.array(
        [
            [
                [[1, 3], [2, 3], [5, 2], [12, 1], [5, 1]],
                [[1, 3], [2, 3], [1, 1], [20, 1], [5, 2]],
                [[1, 3], [2, 3], [9, 1], [4, 1], [5, 2]],
                [[1, 3], [2, 3], [5, 2], [12, 1], [5, 1]],
                [[1, 3], [2, 3], [1, 1], [14, 1], [4, 1]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        ]
    )
    result = np.array([2, 1, 3, 2, 1])
    word = "abcde"
    next_state = state_transition(state, word, result, word_dictionary)
    validation = np.array(
        [
            [
                [[1, 3], [2, 3], [5, 2], [12, 1], [5, 1]],
                [[1, 3], [2, 3], [1, 1], [20, 1], [5, 2]],
                [[1, 3], [2, 3], [9, 1], [4, 1], [5, 2]],
                [[1, 3], [2, 3], [5, 2], [12, 1], [5, 1]],
                [[1, 3], [2, 3], [1, 1], [14, 1], [4, 1]],
                [[1, 2], [2, 1], [3, 3], [4, 2], [5, 1]],
            ]
        ]
    )
    print(next_state)
    assert np.array_equal(next_state, validation)


def test_state_transition2(word_dictionary):
    state = np.array(
        [
            [
                [[1, 3], [2, 3], [5, 2], [12, 1], [5, 1]],
                [[1, 3], [2, 3], [1, 1], [20, 1], [5, 2]],
                [[1, 3], [2, 3], [9, 1], [4, 1], [5, 2]],
                [[1, 3], [2, 3], [5, 2], [12, 1], [5, 1]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        ]
    )
    result = np.array([2, 1, 3, 2, 1])
    word = "abcde"
    next_state = state_transition(state, word, result, word_dictionary)
    validation = np.array(
        [
            [
                [[1, 3], [2, 3], [5, 2], [12, 1], [5, 1]],
                [[1, 3], [2, 3], [1, 1], [20, 1], [5, 2]],
                [[1, 3], [2, 3], [9, 1], [4, 1], [5, 2]],
                [[1, 3], [2, 3], [5, 2], [12, 1], [5, 1]],
                [[1, 2], [2, 1], [3, 3], [4, 2], [5, 1]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        ]
    )
    print(next_state)
    assert np.array_equal(next_state, validation)


def test_reward():
    last_turns = torch.tensor([0])[:, None]
    # shape (1,1)
    rewards = reward_over_states(last_turns)
    validation = torch.zeros((1, 243))
    validation[:, -1] = 1
    print(validation.shape, rewards.shape)
    assert torch.equal(rewards, validation)


def test_rewards():
    last_turns = torch.tensor([0, 1, 1, 0, 0])[:, None]
    # shape (5,1)
    rewards = reward_over_states(last_turns)
    validation = torch.zeros((5, 243))
    validation[:, -1] = 1
    validation[1:3, :-1] = -1
    assert torch.equal(rewards, validation)


def test_reward_given_state():
    state = torch.tensor(
        [
            [
                [[1, 3], [2, 3], [5, 2], [12, 1], [5, 1]],
                [[1, 3], [2, 3], [1, 1], [20, 1], [5, 2]],
                [[1, 3], [2, 3], [9, 1], [4, 1], [5, 2]],
                [[1, 3], [2, 3], [5, 2], [12, 1], [5, 1]],
                [[1, 3], [2, 3], [1, 1], [14, 1], [4, 1]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        ]
    )
    rewards = reward_over_states(state)
    validation = torch.zeros((1, 243)) - 1
    validation[:, -1] = 1
    assert torch.equal(validation, rewards)
