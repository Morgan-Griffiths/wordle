import torch
import os


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
