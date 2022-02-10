from collections import deque
import sys
from unittest import result
import torch
from torch import optim
import numpy as np
from experiments.generate_data import load_data
from experiments.globals import actionSpace, DataTypes, NetworkConfig, dataMapping
from torch.nn.functional import smooth_l1_loss


def train(agent_params, training_params, network_params, dataset):
    net = agent_params["network"](network_params)
    optimizer = optim.Adam(net.parameters(), lr=agent_params["learning_rate"])
    criterion = training_params["criterion"](reduction="sum")
    scores = []
    val_scores = []
    score_window = deque(maxlen=100)
    val_window = deque(maxlen=100)
    for epoch in range(training_params["epochs"]):
        sys.stdout.write("\r")
        optimizer.zero_grad()
        states, actions = dataset["trainX"]
        s_primes, rewards = net(
            torch.as_tensor(states).long(), torch.as_tensor(actions).long()
        )
        target_results, target_rewards = dataset["trainY"]
        result_loss = criterion(s_primes, torch.as_tensor(target_results))
        reward_loss = smooth_l1_loss(
            rewards, torch.as_tensor(target_rewards).unsqueeze(-1).float()
        )
        loss = result_loss + reward_loss
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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="""
        Train and evaluate networks on letter representations.
        """
    )

    parser.add_argument(
        "-d",
        "--datatype",
        default=DataTypes.WORDLE,
        type=str,
        metavar=f"[{DataTypes.LETTERS},{DataTypes.CAPITALS},{DataTypes.WORDLE},{DataTypes.MULTI_TARGET},{DataTypes.CONSTELLATION}]",
        help="Which dataset to train on",
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of training epochs", default=500, type=int
    )
    parser.add_argument(
        "--resume", help="resume training from an earlier run", action="store_true"
    )
    parser.add_argument(
        "-lr", help="resume training from an earlier run", type=float, default=0.003
    )
    parser.set_defaults(resume=False)

    args = parser.parse_args()
    network_path = "weights/represent"
    loss_type = dataMapping[args.datatype]
    agent_params = {
        "learning_rate": args.lr,
        "network": NetworkConfig.DataModels[args.datatype],
        "save_dir": "checkpoints",
        "save_path": network_path,
        "load_path": network_path,
    }
    training_params = {
        "resume": args.resume,
        "epochs": args.epochs,
        "criterion": NetworkConfig.LossFunctions[loss_type],
        "network": NetworkConfig.DataModels[args.datatype],
        "load_path": network_path,
    }
    network_params = {
        "seed": 346,
        "nA": actionSpace[args.datatype],
        "load_path": network_path,
        "emb_size": 16,
    }
    dataset = load_data(args.datatype)
    train(agent_params, training_params, network_params, dataset)
