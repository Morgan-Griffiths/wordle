from MCTS_mu import MCTS, MCTS_dict, Node
from wordle import Wordle
import torch
from globals import Embeddings, result_index_dict, index_result_dict
from utils import result_from_state
import time

# def test_mcts(env, mcts: MCTS, mu_agent):
#     state, reward, done = env.reset()
#     print(env.word)
#     root = Node(None, 1)
#     root.state = torch.as_tensor(state).long().unsqueeze(0)
#     root.reward = reward
#     mcts.run(root, mu_agent)
#     assert root.value != 0


def test_mcts_equality(env, mcts: MCTS, mcts_dict: MCTS_dict, mu_agent, config):
    state, reward, done = env.reset()
    turn = 0
    tic = time.time()
    mcts_info = mcts_dict.run(mu_agent, state, turn)
    print(f"mcts_dict run {time.time() - tic}")
    print(mcts_info)
    assert mcts_info["max_tree_depth"] == 6
    tic = time.time()
    root, mcts_info = mcts.run(mu_agent, state, reward, turn)
    print(f"mcts run {time.time() - tic}")
    print(mcts_info)
    asdf

    # root, mcts_info = MCTS_dict.run(mu_agent, state, reward, turn)
    # assert root.visit_count == config.num_simulations
    # assert mcts_info["max_tree_depth"] == 6
