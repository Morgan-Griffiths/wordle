from MCTS import MCTS


def test_mcts(env, mcts: MCTS):
    mcts.search(env)
