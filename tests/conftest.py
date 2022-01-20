from ast import Mod
import pytest
from wordle import Wordle
from ML.networks import Policy, Q_learning
from globals import Dims, State, Models
from numpy import zeros
from ML.agent import Agent


@pytest.fixture
def env():
    return Wordle()


@pytest.fixture
def agent():
    return Agent(Dims.OUTPUT, 1234, model=Models.AC_LEARNING)


@pytest.fixture
def q_agent():
    return Agent(Dims.OUTPUT, 1234, model=Models.Q_LEARNING)


@pytest.fixture
def q_network():
    seed = 1234
    return Q_learning(seed, Dims.OUTPUT)


@pytest.fixture
def policy():
    seed = 1234
    return Policy(seed, Dims.OUTPUT)


@pytest.fixture
def fake_input():
    return zeros(State.SHAPE)[None, :]
