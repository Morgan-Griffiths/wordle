from MCTS_mu import MCTS, Node
from MCTS_optimized import MCTS_dict
import torch
from globals import Embeddings
from utils import result_from_state

# def test_mcts(env, mcts: MCTS, mu_agent):
#     state, reward, done = env.reset()
#     print(env.word)
#     root = Node(None, 1)
#     root.state = torch.as_tensor(state).long().unsqueeze(0)
#     root.reward = reward
#     mcts.run(root, mu_agent)
#     assert root.value != 0


# def test_mcts_equality(env, mcts: MCTS, mcts_dict: MCTS_dict, mu_agent, config, mappings):
#     state, reward, done = env.reset()
#     print(env.word)
#     turn = 0
#     action = 5
#     env.step(env.action_to_string(action))
#     result = state[turn][:, Embeddings.RESULT]
#     print(env.action_to_string(action))
#     res = result_from_state(turn,state)
#     print(res)
#     asdf
# root, mcts_info = mcts.run(mu_agent, state, reward, turn)
# assert root.visit_count == config.num_simulations
# assert mcts_info["max_tree_depth"] == 6

# root, mcts_info = MCTS_dict.run(mu_agent, state, reward, turn)
# assert root.visit_count == config.num_simulations
# assert mcts_info["max_tree_depth"] == 6
