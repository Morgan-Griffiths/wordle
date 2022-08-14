import pathlib
import pickle
import torch
import os
from collections import OrderedDict


def load_replay_buffer():
    try:
        replay_buffer_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "replay_buffer_storage"
            / "replay_buffer.pkl"
        )
        with open(replay_buffer_path, "rb") as f:
            replay_buffer_infos = pickle.load(f)
    except:
        # Configure running options
        options = ["Specify paths manually"] + sorted(
            (pathlib.Path("replay_buffer_storage")).glob("*/")
        )
        options.reverse()
        print()
        for i in range(len(options)):
            print(f"{i}. {options[i]}")

        choice = input("Enter a number to choose a dataset to load: ")
        valid_inputs = [str(i) for i in range(len(options))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)

        if choice == (len(options) - 1):
            # manual path option
            replay_buffer_path = input(
                "Enter a path to the replay_buffer.pkl, or ENTER if none: "
            )
            while replay_buffer_path and not pathlib.Path(replay_buffer_path).is_file():
                replay_buffer_path = input("Invalid replay buffer path. Try again: ")
        else:
            replay_buffer_path = (
                pathlib.Path(__file__).resolve().parents[0] / options[choice]
            )
            if not replay_buffer_path.is_file():
                replay_buffer_path = None
        with open(replay_buffer_path, "rb") as f:
            replay_buffer_infos = pickle.load(f)
    return replay_buffer_infos


def load_model_menu(muzero):
    # Configure running options
    options = ["Specify paths manually"] + sorted((pathlib.Path("weights")).glob("*/"))
    options.reverse()
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    if choice == (len(options) - 1):
        # manual path option
        checkpoint_path = input(
            "Enter a path to the model.checkpoint, or ENTER if none: "
        )
        while checkpoint_path and not pathlib.Path(checkpoint_path).is_file():
            checkpoint_path = input("Invalid checkpoint path. Try again: ")
        replay_buffer_path = input(
            "Enter a path to the replay_buffer.pkl, or ENTER if none: "
        )
        while replay_buffer_path and not pathlib.Path(replay_buffer_path).is_file():
            replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        checkpoint_path = options[choice] / "model.checkpoint"
        # replay_buffer_path = options[choice] / "replay_buffer.pkl"
        replay_buffer_path = (
            pathlib.Path(__file__).resolve().parents[0]
            / "replay_buffer_storage"
            / "replay_buffer.pkl"
        )
        if not replay_buffer_path.is_file():
            replay_buffer_path = None

    muzero.load_model(
        checkpoint_path=checkpoint_path,
        replay_buffer_path=replay_buffer_path,
    )


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
        if k.find("module") > -1:
            name = "".join(k.split(".module"))
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def add_module(path):
    state_dict = load_path(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = "module." + k
        new_state_dict[name] = v
    return new_state_dict


def is_path_ddp(path):
    is_ddp = False
    state_dict = load_path(path)
    for k in state_dict.keys():
        if k.find("module.") > -1:
            is_ddp = True
        break
    return is_ddp


def is_net_ddp(net):
    is_ddp = False
    for name, param in net.named_parameters():
        if name.find("module.") > -1:
            is_ddp = True
        break


def load_path(path):
    if torch.cuda.is_available():
        state_dict = torch.load(path)
    else:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
    return state_dict
