import pytest
from ML.agents.q_agent import Q_agent
from config import Config
from wordle import Wordle
from ML.networks import Policy, Q_learning
from globals import AgentData, Dims, State
from numpy import zeros
from ML.agents.base_agent import Agent
from ML.agents.ppo import PPO_agent
from MCTS import MCTS

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
def network_params():
    return {"nA": Dims.OUTPUT}


@pytest.fixture
def agent():
    return Agent(Dims.OUTPUT, 1234)


@pytest.fixture
def q_agent(network_params, config):
    return Q_agent(network_params, config)


@pytest.fixture
def ppo_agent(network_params, config):
    return PPO_agent(network_params, config)


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


@pytest.fixture
def mcts(env, ppo_agent):
    return MCTS(env, ppo_agent)
