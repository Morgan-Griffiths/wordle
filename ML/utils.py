import torch
import os
from collections import OrderedDict 


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

def strip_module(state_dict):
    # state_dict = load_path(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.find('module') > -1:
            name = ''.join(k.split('.module'))
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def add_module(path):
    state_dict = load_path(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.'+k
        new_state_dict[name] = v
    return new_state_dict

def is_path_ddp(path):
    is_ddp = False
    state_dict = load_path(path)
    for k in state_dict.keys():
        if k.find('module.') > -1:
            is_ddp = True
        break
    return is_ddp

def is_net_ddp(net):
    is_ddp = False
    for name,param in net.named_parameters():
        if name.find('module.') > -1:
            is_ddp = True
        break
    return is_ddp
