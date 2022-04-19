import torch
import os


def reward_over_states(last_turn: torch.tensor):
    """Returns rewards over next states"""
    B = last_turn.shape[0]
    rewards = torch.zeros((B, 243))
    last_mask = torch.where(last_turn == 1)[0]
    rewards[:, -1] = 1
    rewards[last_mask, :-1] = -1
    return rewards


def load_weights(network, path):
    network.load_state_dict(torch.load(path))
    network.eval()
    return network


def save_weights(network, path):
    print(path)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(network.state_dict(), path)


def hard_update(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
