import torch
import numpy as np
from globals import AgentData, Outputs
from prettytable import PrettyTable


def return_rewards(turn: int, reward: float):
    sign = -1 if reward < 0 else 1
    return list(
        reversed([torch.Tensor([reward - (0.05 * num) * sign]) for num in range(turn)])
    )


def to_tensor(state):
    return torch.as_tensor(state, dtype=torch.int32)


def store_state(
    data_params: dict,
    state: np.array,
    done: bool,
) -> dict:
    data_params[AgentData.STATES].append(to_tensor(state))
    data_params[AgentData.DONES].append(to_tensor(done))
    return data_params


def store_outputs(data_params: dict, outputs: dict) -> dict:
    data_params[AgentData.ACTION_PROBS].append(outputs[Outputs.ACTION_PROB])
    data_params[AgentData.VALUES].append(outputs[Outputs.VALUES])
    data_params[AgentData.ACTIONS].append(outputs[Outputs.ACTION])
    return data_params


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
