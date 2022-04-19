import pytest
from config import Config
from wordle import Wordle
from ML.networks import MuZeroNet
from globals import AgentData, Dims, State
from numpy import zeros
from MCTS_mu import MCTS


@pytest.fixture
def network_params():
    return {
        "seed": 346,
        "nA": Dims.OUTPUT,
        "load_path": "mu_zero",
        "emb_size": 16,
    }


@pytest.fixture
def data_params():
    return {
        AgentData.STATES: [],
        AgentData.ACTIONS: [],
        AgentData.ACTION_PROBS: [],
        AgentData.ACTION_PROB: [],
        AgentData.VALUES: [],
        AgentData.REWARDS: [],
        AgentData.DONES: [],
    }


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def env():
    return Wordle()


@pytest.fixture
def fake_input():
    return zeros(State.SHAPE)[None, :]


@pytest.fixture
def mu_agent():
    return MuZeroNet(5)


@pytest.fixture
def mcts(config):
    return MCTS(config)
