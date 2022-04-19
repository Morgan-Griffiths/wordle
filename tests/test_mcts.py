from MCTS_mu import MCTS, Node
from wordle import Wordle
import torch


def test_mcts(env, mcts: MCTS, mu_agent):
    state, reward, done = env.reset()
    print(env.word)
    root = Node(None, 1)
    root.state = torch.as_tensor(state).long().unsqueeze(0)
    root.reward = reward
    mcts.run(root, mu_agent)
    assert root.value != 0
