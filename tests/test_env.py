from tokenize import Token
from globals import Embeddings, Mappings, Tokens, State
import numpy as np

from wordle import Wordle


def test_init(env: Wordle):
    assert len(env.word) == 5
    assert env.turn == 0
    assert len(env.alphabet) == 27
    assert isinstance(env.alphabet, dict)
    assert isinstance(env.word, str)


def test_state(env: Wordle, mappings: Mappings):
    env.word = "raphe"
    assert np.array_equal(env.state, np.zeros(State.SHAPE, dtype=np.int8))
    state, rewards, done = env.step("hello")
    comparison_state = np.zeros(State.SHAPE, dtype=np.int8)
    turn = 0
    comparison_state[turn, 0, Embeddings.LETTER] = mappings.alphabet_dict["h"]
    comparison_state[turn, 0, Embeddings.RESULT] = Tokens.CONTAINED
    comparison_state[turn, 1, Embeddings.LETTER] = mappings.alphabet_dict["e"]
    comparison_state[turn, 1, Embeddings.RESULT] = Tokens.CONTAINED
    comparison_state[turn, 2, Embeddings.LETTER] = mappings.alphabet_dict["l"]
    comparison_state[turn, 2, Embeddings.RESULT] = Tokens.MISSING
    comparison_state[turn, 3, Embeddings.LETTER] = mappings.alphabet_dict["l"]
    comparison_state[turn, 3, Embeddings.RESULT] = Tokens.MISSING
    comparison_state[turn, 4, Embeddings.LETTER] = mappings.alphabet_dict["o"]
    comparison_state[turn, 4, Embeddings.RESULT] = Tokens.MISSING
    assert np.array_equal(env.state, comparison_state)
    state, rewards, done = env.step("wrath")
    turn = 1
    comparison_state[turn, 0, Embeddings.LETTER] = mappings.alphabet_dict["w"]
    comparison_state[turn, 0, Embeddings.RESULT] = Tokens.MISSING
    comparison_state[turn, 1, Embeddings.LETTER] = mappings.alphabet_dict["r"]
    comparison_state[turn, 1, Embeddings.RESULT] = Tokens.CONTAINED
    comparison_state[turn, 2, Embeddings.LETTER] = mappings.alphabet_dict["a"]
    comparison_state[turn, 2, Embeddings.RESULT] = Tokens.CONTAINED
    comparison_state[turn, 3, Embeddings.LETTER] = mappings.alphabet_dict["t"]
    comparison_state[turn, 3, Embeddings.RESULT] = Tokens.MISSING
    comparison_state[turn, 4, Embeddings.LETTER] = mappings.alphabet_dict["h"]
    comparison_state[turn, 4, Embeddings.RESULT] = Tokens.CONTAINED
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
    env.word = "hello"
    _, rewards, done = env.step("hello")
    assert rewards == 1
    env.reset()
    env.word = "hello"
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("hello")
    assert rewards == 1
    env.reset()
    env.word = "hello"
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("hello")
    assert rewards == 1
    env.reset()
    env.word = "hello"
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("hello")
    assert rewards == 1
    env.reset()
    env.word = "hello"
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("hello")
    assert rewards == 1
    env.reset()
    env.word = "hello"
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("world")
    _, rewards, done = env.step("hello")
    assert rewards == 1


def test_success(env):
    env.word = "hello"
    _, rewards, done = env.step("shire")
    assert done == False
    assert rewards == 0
    assert env.turn == 1
    assert env.alphabet["s"] == 1
    assert env.alphabet["h"] == 2
    assert env.alphabet["i"] == 1
    assert env.alphabet["r"] == 1
    assert env.alphabet["e"] == 2
    _, rewards, done = env.step("happy")
    assert done == False
    assert rewards == 0
    assert env.turn == 2
    assert env.alphabet["h"] == 3
    assert env.alphabet["a"] == 1
    assert env.alphabet["p"] == 1
    assert env.alphabet["p"] == 1
    assert env.alphabet["y"] == 1
    _, rewards, done = env.step("zebra")
    assert done == False
    assert rewards == 0
    assert env.turn == 3
    assert env.alphabet["z"] == 1
    assert env.alphabet["e"] == 3
    assert env.alphabet["b"] == 1
    assert env.alphabet["r"] == 1
    assert env.alphabet["a"] == 1
    _, rewards, done = env.step("world")
    assert done == False
    assert rewards == 0
    assert env.turn == 4
    assert env.alphabet["w"] == 1
    assert env.alphabet["o"] == 2
    assert env.alphabet["r"] == 1
    assert env.alphabet["l"] == 3
    assert env.alphabet["d"] == 1
    _, rewards, done = env.step("hello")
    assert done == True
    assert rewards == 1
    assert env.turn == 5
    assert env.alphabet["h"] == 3
    assert env.alphabet["e"] == 3
    assert env.alphabet["l"] == 3
    assert env.alphabet["l"] == 3
    assert env.alphabet["o"] == 3


def test_fail(env):
    env.word = "hello"
    _, rewards, done = env.step("shire")
    assert done == False
    assert rewards == 0
    assert env.turn == 1
    _, rewards, done = env.step("happy")
    assert done == False
    assert rewards == 0
    assert env.turn == 2
    _, rewards, done = env.step("zebra")
    assert done == False
    assert rewards == 0
    assert env.turn == 3
    _, rewards, done = env.step("world")
    assert done == False
    assert rewards == 0
    assert env.turn == 4
    _, rewards, done = env.step("world")
    assert done == False
    assert rewards == 0
    assert env.turn == 5
    _, rewards, done = env.step("world")
    assert done == True
    assert rewards == -1
    assert env.turn == 6
