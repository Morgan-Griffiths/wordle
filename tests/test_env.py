from tokenize import Token
from globals import Tokens, alphabet_dict, State
import numpy as np


def test_init(env):
    assert len(env.word) == 5
    assert env.turn == 0
    assert len(env.alphabet) == 26
    assert isinstance(env.alphabet, dict)
    assert isinstance(env.word, str)


def test_state(env):
    env.word = "RAPHE"
    assert np.array_equal(env.state, np.zeros(State.SHAPE))
    state, rewards, done = env.step("HELLO")
    comparison_state = np.zeros(State.SHAPE)
    turn = 0
    comparison_state[turn, 0, alphabet_dict["H"]] = Tokens.CONTAINED
    comparison_state[turn, 1, alphabet_dict["E"]] = Tokens.CONTAINED
    comparison_state[turn, 2, alphabet_dict["L"]] = Tokens.MISSING
    comparison_state[turn, 3, alphabet_dict["L"]] = Tokens.MISSING
    comparison_state[turn, 4, alphabet_dict["O"]] = Tokens.MISSING
    assert np.array_equal(env.state, comparison_state)
    state, rewards, done = env.step("RAPER")
    turn = 1
    comparison_state[turn, 0, alphabet_dict["R"]] = Tokens.EXACT
    comparison_state[turn, 1, alphabet_dict["A"]] = Tokens.EXACT
    comparison_state[turn, 2, alphabet_dict["P"]] = Tokens.EXACT
    comparison_state[turn, 3, alphabet_dict["E"]] = Tokens.CONTAINED
    comparison_state[turn, 4, alphabet_dict["R"]] = Tokens.MISSING
    # print("comparison_state", np.where(comparison_state == 1))
    # print("env.state", np.where(env.state == 1))
    assert np.array_equal(env.state, comparison_state)


def test_results(env):
    env.word = "RAPHE"
    results = env.evaluate_word("RAPER")
    assert np.array_equal(results, np.array([2.0, 2.0, 2.0, 1.0, 0.0]))
    env.reset()
    env.word = "SASSY"
    results = env.evaluate_word("SHIPS")
    assert np.array_equal(results, np.array([2.0, 0.0, 0.0, 0.0, 1.0]))


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
    assert env.alphabet["S"] == 0
    assert env.alphabet["H"] == 1
    assert env.alphabet["I"] == 0
    assert env.alphabet["R"] == 0
    assert env.alphabet["E"] == 1
    _, rewards, done = env.step("HAPPY")
    assert done == False
    assert rewards == 0
    assert env.turn == 2
    assert env.alphabet["H"] == 2
    assert env.alphabet["A"] == 0
    assert env.alphabet["P"] == 0
    assert env.alphabet["P"] == 0
    assert env.alphabet["Y"] == 0
    _, rewards, done = env.step("WORDS")
    assert done == False
    assert rewards == 0
    assert env.turn == 3
    assert env.alphabet["W"] == 0
    assert env.alphabet["O"] == 1
    assert env.alphabet["R"] == 0
    assert env.alphabet["D"] == 0
    assert env.alphabet["S"] == 0
    _, rewards, done = env.step("WORLD")
    assert done == False
    assert rewards == 0
    assert env.turn == 4
    assert env.alphabet["W"] == 0
    assert env.alphabet["O"] == 1
    assert env.alphabet["R"] == 0
    assert env.alphabet["L"] == 2
    assert env.alphabet["D"] == 0
    _, rewards, done = env.step("Hello")
    assert done == True
    assert rewards == 0.80
    assert env.turn == 5
    assert env.alphabet["H"] == 2
    assert env.alphabet["E"] == 2
    assert env.alphabet["L"] == 2
    assert env.alphabet["L"] == 2
    assert env.alphabet["O"] == 2


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
