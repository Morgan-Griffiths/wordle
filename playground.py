import torch.nn as nn
import torch
import numpy as np
from globals import State
from utils import to_tensor
from itertools import permutations
from MCTS import MCTS
from wordle import Wordle


env = Wordle()
env.step("HELLO")
env.visualize_state()
turn = np.min(np.where(env.state[:, 0, 0] == 0)[0])
print(turn, env.turn)
# print(env.state)
