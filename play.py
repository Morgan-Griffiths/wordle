from MCTS_mu import MCTS, MCTS_dict
from MCTS_numpy import MCTS_numpy
from MCTS_optimized import MCTSNode

import torch
import numpy as np
from ML.networks import MuZeroNet
from config import Config


from globals import DynamicOutputs, PolicyOutputs, index_result_dict, target_dictionary
from utils import state_transition
from wordle import Wordle
import time

config = Config()
env = Wordle(word_restriction=config.word_restriction)
config.word_to_index = env.dictionary_word_to_index
config.index_to_word = env.dictionary_index_to_word
mc = MCTS_dict(config)
mcts = MCTS(config)
state, reward, done = env.reset()
# state, reward, done = env.step(env.action_to_string(config.action_space))
# print("state", state)
# print(len(config.index_to_word.keys()))
# print(target_dictionary)
# print(index_result_dict)
# print(env.dictionary_index_to_word[0])
# print(config.action_space)
# print(len(env.dictionary_index_to_word.keys()))
# print(len(target_dictionary))
turn = 0
agent = MuZeroNet(config)
tic = time.time()
root, info = mc.run(agent, state, reward, turn)
# info = mc.run(agent, state, turn)
print(info)
print(f"took {time.time() - tic} seconds")
