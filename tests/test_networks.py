import torch
from ML.agent import Agent
from globals import AgentData, Dims
from wordle import Wordle
from utils import to_tensor, return_rewards, store_state, store_outputs


def test_input(policy, fake_input):
    action, prob, _, value = policy(to_tensor(fake_input))
    # print(value.shape, value, type(value))
    # assert prob.shape[-1] == 1
    assert value.shape[-1] == Dims.OUTPUT


def test_learning(env: Wordle, agent: Agent):
    env.word = "HELLO"
    data_params = {
        AgentData.STATES: [],
        AgentData.ACTIONS: [],
        AgentData.ACTION_PROBS: [],
        AgentData.VALUES: [],
        AgentData.REWARDS: [],
    }
    word = "RAPHE"
    state, reward, done = env.step(word)
    data_params = store_state(data_params, state=state)
    outputs:dict = agent(state)
    data_params = store_outputs(data_params, outputs)
    rewards = return_rewards(env.turn, reward)
    data_params[AgentData.REWARDS] = rewards
    agent.learn(data_params)
    new_outputs = agent(state)
    print(new_outputs['values'].shape, outputs['values'].shape)
    print(new_outputs['values'][:, outputs['action']], outputs['values'][:, outputs['action']])
    assert new_outputs['values'][:, outputs['action']] < outputs['values'][:, outputs['action']]


def test_full_learning_loop(env: Wordle, agent: Agent):
    data_params = {
        AgentData.STATES: [],
        AgentData.ACTIONS: [],
        AgentData.ACTION_PROBS: [],
        AgentData.VALUES: [],
        AgentData.REWARDS: [],
    }
    state, reward, done = env.reset()
    data_params = store_state(data_params, state=state)
    env.word = "HELLO"
    outputs:dict = agent(state)
    data_params = store_outputs(data_params, outputs)
    word = "RAPHE"
    state, reward, done = env.step(word)
    data_params = store_state(data_params, state=state)
    outputs:dict = agent(state)
    data_params = store_outputs(data_params, outputs)
    word = "HOLDS"
    state, reward, done = env.step(word)
    data_params = store_state(data_params, state=state)
    outputs:dict = agent(state)
    data_params = store_outputs(data_params, outputs)
    word = "HELLO"
    state, reward, done = env.step(word)
    rewards = return_rewards(env.turn, reward)
    data_params[AgentData.REWARDS] = rewards
    agent.learn(data_params)
