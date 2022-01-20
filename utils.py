import torch
import numpy as np
from globals import AgentData


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
) -> dict:
    data_params[AgentData.STATES].append(to_tensor(state))
    return data_params


def store_outputs(data_params: dict, outputs: dict) -> dict:
    data_params[AgentData.ACTION_PROBS].append(outputs["prob"])
    data_params[AgentData.VALUES].append(outputs["value"])
    data_params[AgentData.ACTIONS].append(outputs["action"])
    return data_params
