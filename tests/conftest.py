import pytest
from config import Config
from wordle import Wordle
from ML.networks import MuZeroNet
from globals import AgentData, Dims, State, Mappings
from numpy import zeros
from MCTS_mu import MCTS
from MCTS_optimized import MCTS_dict


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
def mappings():
    return Mappings(None)


@pytest.fixture
def env(mappings):
    return Wordle(mappings)


@pytest.fixture
def config():
    config = Config()
    return config


@pytest.fixture
def fake_input():
    return zeros(State.SHAPE)[None, :]


@pytest.fixture
def mu_agent(config: Config):
    mappings = Mappings(config.word_restriction)
    return MuZeroNet(config, mappings)


@pytest.fixture
def mcts(config):
    return MCTS(config)


@pytest.fixture
def mcts_dict(config):
    return MCTS_dict(config)


@pytest.fixture
def mcts_opt(config):
    return MCTS_dict(config)
