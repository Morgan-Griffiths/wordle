from tokenize import Token
from globals import Embeddings, Results, Tokens, alphabet_dict, State
import numpy as np

from wordle import Wordle


def test_init(env):
    assert len(env.word) == 5
    assert env.turn == 0
    assert len(env.alphabet) == 26
    assert isinstance(env.alphabet, dict)
    assert isinstance(env.word, str)


def test_state(env: Wordle):
    env.word = "RAPHE"
    assert np.array_equal(env.state, np.zeros(State.SHAPE))
    state, rewards, done = env.step("HELLO")
    comparison_state = np.zeros(State.SHAPE)
    turn = 0
    comparison_state[turn, 0, Embeddings.LETTER] = alphabet_dict["H"]
    comparison_state[turn, 0, Embeddings.RESULT] = Tokens.CONTAINED
    comparison_state[turn, 1, Embeddings.LETTER] = alphabet_dict["E"]
    comparison_state[turn, 1, Embeddings.RESULT] = Tokens.CONTAINED
    comparison_state[turn, 2, Embeddings.LETTER] = alphabet_dict["L"]
    comparison_state[turn, 2, Embeddings.RESULT] = Tokens.MISSING
    comparison_state[turn, 3, Embeddings.LETTER] = alphabet_dict["L"]
    comparison_state[turn, 3, Embeddings.RESULT] = Tokens.MISSING
    comparison_state[turn, 4, Embeddings.LETTER] = alphabet_dict["O"]
    comparison_state[turn, 4, Embeddings.RESULT] = Tokens.MISSING
    comparison_state[turn, :, Embeddings.WORD] = env.dictionary_word_to_index["HELLO"]
    assert np.array_equal(env.state, comparison_state)
    state, rewards, done = env.step("RAPER")
    turn = 1
    comparison_state[turn, 0, Embeddings.LETTER] = alphabet_dict["R"]
    comparison_state[turn, 0, Embeddings.RESULT] = Tokens.EXACT
    comparison_state[turn, 1, Embeddings.LETTER] = alphabet_dict["A"]
    comparison_state[turn, 1, Embeddings.RESULT] = Tokens.EXACT
    comparison_state[turn, 2, Embeddings.LETTER] = alphabet_dict["P"]
    comparison_state[turn, 2, Embeddings.RESULT] = Tokens.EXACT
    comparison_state[turn, 3, Embeddings.LETTER] = alphabet_dict["E"]
    comparison_state[turn, 3, Embeddings.RESULT] = Tokens.CONTAINED
    comparison_state[turn, 4, Embeddings.LETTER] = alphabet_dict["R"]
    comparison_state[turn, 4, Embeddings.RESULT] = Tokens.MISSING
    comparison_state[turn, :, Embeddings.WORD] = env.dictionary_word_to_index["RAPER"]
    assert np.array_equal(env.state, comparison_state)


# def test_results(env):
#     env.word = "RAPHE"
#     results = env.evaluate_word("RAPER")
#     assert np.array_equal(results, np.array([3, 3, 3, 2, 1]))
#     env.reset()
#     env.word = "SASSY"
#     results = env.evaluate_word("SHIPS")
#     assert np.array_equal(results, np.array([2, 0, 0, 0, 1]))


def test_rewards(env):
    env.word = "HELLO"
    _, rewards, done = env.step("HELLO")
    assert rewards == 1
    env.reset()
    env.word = "HELLO"
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("HELLO")
    assert rewards == 0.95
    env.reset()
    env.word = "HELLO"
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("HELLO")
    assert rewards == 0.90
    env.reset()
    env.word = "HELLO"
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("HELLO")
    assert rewards == 0.85
    env.reset()
    env.word = "HELLO"
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("HELLO")
    assert rewards == 0.80
    env.reset()
    env.word = "HELLO"
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("WORLD")
    _, rewards, done = env.step("HELLO")
    assert rewards == 0.75


def test_success(env):
    env.word = "HELLO"
    _, rewards, done = env.step("SHIRE")
    assert done == False
    assert rewards == 0
    assert env.turn == 1
    assert env.alphabet["S"] == 1
    assert env.alphabet["H"] == 2
    assert env.alphabet["I"] == 1
    assert env.alphabet["R"] == 1
    assert env.alphabet["E"] == 2
    _, rewards, done = env.step("HAPPY")
    assert done == False
    assert rewards == 0
    assert env.turn == 2
    assert env.alphabet["H"] == 3
    assert env.alphabet["A"] == 1
    assert env.alphabet["P"] == 1
    assert env.alphabet["P"] == 1
    assert env.alphabet["Y"] == 1
    _, rewards, done = env.step("WORDS")
    assert done == False
    assert rewards == 0
    assert env.turn == 3
    assert env.alphabet["W"] == 1
    assert env.alphabet["O"] == 2
    assert env.alphabet["R"] == 1
    assert env.alphabet["D"] == 1
    assert env.alphabet["S"] == 1
    _, rewards, done = env.step("WORLD")
    assert done == False
    assert rewards == 0
    assert env.turn == 4
    assert env.alphabet["W"] == 1
    assert env.alphabet["O"] == 2
    assert env.alphabet["R"] == 1
    assert env.alphabet["L"] == 3
    assert env.alphabet["D"] == 1
    _, rewards, done = env.step("Hello")
    assert done == True
    assert rewards == 0.80
    assert env.turn == 5
    assert env.alphabet["H"] == 3
    assert env.alphabet["E"] == 3
    assert env.alphabet["L"] == 3
    assert env.alphabet["L"] == 3
    assert env.alphabet["O"] == 3


def test_fail(env):
    env.word = "HELLO"
    _, rewards, done = env.step("SHIRE")
    assert done == False
    assert rewards == 0
    assert env.turn == 1
    _, rewards, done = env.step("HAPPY")
    assert done == False
    assert rewards == 0
    assert env.turn == 2
    _, rewards, done = env.step("WORDS")
    assert done == False
    assert rewards == 0
    assert env.turn == 3
    _, rewards, done = env.step("WORLD")
    assert done == False
    assert rewards == 0
    assert env.turn == 4
    _, rewards, done = env.step("WORLD")
    assert done == False
    assert rewards == 0
    assert env.turn == 5
    _, rewards, done = env.step("WORLD")
    assert done == True
    assert rewards == -1
    assert env.turn == 6
