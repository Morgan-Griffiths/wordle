from collections import deque
import numpy as np
import sys
import copy
import ray

import torch.nn.functional as F
from torch.nn import SmoothL1Loss
from torch import optim
import torch
from ML.utils import load_replay_buffer

from ray_files.replay_buffer import ReplayBuffer
from globals import Mappings, PolicyOutputs, CHECKPOINT
from ML.networks import ZeroPolicy, MuZeroNet
from config import Config


def test_policy(agent_params, training_params, config, per_buffer):
    next_batch = per_buffer.get_batch.remote()
    index_batch, batch = ray.get(next_batch)
    (
        state_batch,
        action_batch,
        value_batch,
        reward_batch,
        policy_batch,
        weight_batch,
        result_batch,
        word_batch,
        gradient_scale_batch,
    ) = batch
    net = ZeroPolicy(config)
    value_criterion = SmoothL1Loss()
    if training_params["resume"]:
        net.load_state_dict(torch.load(training_params["load_path"]))
    optimizer = optim.AdamW(net.parameters(), lr=agent_params["learning_rate"])
    scores = []
    score_window = deque(maxlen=100)
    for epoch in range(training_params["epochs"]):
        sys.stdout.write("\r")
        optimizer.zero_grad()
        outputs: PolicyOutputs = net(state_batch)
        policy_loss = (
            F.kl_div(outputs.logprobs, policy_batch, reduction="none")
            .sum(dim=1)
            .unsqueeze(1)
        )
        value_loss = value_criterion(outputs.value, reward_batch)
        loss = (policy_loss + value_loss).sum()
        loss.backward()
        optimizer.step()
        score_window.append(loss.item())
        scores.append(np.mean(score_window))
        sys.stdout.flush()
        sys.stdout.write(f", epoch {epoch}")
        sys.stdout.flush()
        sys.stdout.write(f", loss {np.mean(score_window):.4f}")
        sys.stdout.flush()
    print(f"Saving weights to {training_params['load_path']}")
    torch.save(net.state_dict(), training_params["load_path"])
    validation(net, per_buffer)


def validation(network, per_buffer):
    next_batch = per_buffer.get_batch.remote()
    index_batch, batch = ray.get(next_batch)
    (
        state_batch,
        action_batch,
        value_batch,
        reward_batch,
        policy_batch,
        weight_batch,
        result_batch,
        word_batch,
        gradient_scale_batch,
    ) = batch
    while True:
        print(f"Number of states {len(state_batch)}")
        valid_inputs = [str(i) for i in (range(len(state_batch)))]
        choice = input(f"Pick a number between 0-{len(state_batch)}")
        while choice not in valid_inputs:
            print("invalid input")
            choice = input(f"Pick a number between 0-{len(state_batch)}")
        sample_idx = int(choice)
        # sample_idx = np.random.choice(len(states))
        state = state_batch[sample_idx][None, :]
        target_action = word_batch[sample_idx]
        target_reward = reward_batch[sample_idx]

        state = torch.as_tensor(state).long()
        print("state", state, state.shape)
        print("target_action", target_action)
        print("target_reward", target_reward)
        with torch.no_grad():
            policy_outputs: PolicyOutputs = network(state)
            print("action", policy_outputs.action)
            print("value", policy_outputs.value)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="""
        Train and evaluate networks on letter representations.
        """
    )

    parser.add_argument(
        "-e", "--epochs", help="Number of training epochs", default=100, type=int
    )
    parser.add_argument(
        "--resume", help="resume training from an earlier run", action="store_true"
    )
    parser.add_argument(
        "-lr", help="resume training from an earlier run", type=float, default=3e-3
    )
    parser.add_argument(
        "-b",
        dest="batch_size",
        help="batch size",
        type=int,
        default=4,
    )
    parser.set_defaults(resume=False)
    parser.set_defaults(validate=False)

    args = parser.parse_args()
    config = Config()
    print(args)
    network_path = "weights/policy_test"
    agent_params = {
        "learning_rate": args.lr,
        "network": ZeroPolicy,
        "save_dir": "checkpoints",
        "save_path": network_path,
        "load_path": network_path,
    }
    training_params = {
        "resume": args.resume,
        "epochs": args.epochs,
        "network": ZeroPolicy,
        "load_path": network_path,
    }
    network_params = {
        "seed": 346,
        "load_path": network_path,
        "emb_size": 16,
    }
    config.batch_size = args.batch_size
    mappings = Mappings(config.word_restriction)
    buffer_info = load_replay_buffer()
    checkpoint = copy.copy(CHECKPOINT)
    checkpoint["num_played_steps"] = buffer_info["num_played_steps"]
    checkpoint["num_played_games"] = buffer_info["num_played_games"]
    checkpoint["num_reanalysed_games"] = buffer_info["num_reanalysed_games"]
    per_buffer = ReplayBuffer.remote(checkpoint, buffer_info["buffer"], config)
    test_policy(agent_params, training_params, config, per_buffer)
