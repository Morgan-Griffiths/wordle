import numpy as np
import sys
from collections import deque
from torch import optim
import torch

from experiments.generate_data import load_data
from experiments.dataloader import return_dataloader
from experiments.globals import actionSpace, DataTypes, NetworkConfig, dataMapping


def train(data_dict, agent_params, training_params, dataset):
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
        outputs = net(torch.as_tensor(dataset["trainX"]).long())
        loss = criterion(outputs, torch.as_tensor(dataset["trainY"]).squeeze(-1))
        loss.backward()
        optimizer.step()
        # for i, data in enumerate(data_dict["trainloader"], 1):
        #     sys.stdout.write("\r")
        #     # get the inputs; data is a list of [inputs, targets]
        #     inputs, targets = data.values()
        #     # zero the parameter gradients
        #     optimizer.zero_grad()
        #     outputs = net(inputs)
        #     loss = criterion(outputs, targets.squeeze(-1))
        #     loss.backward()
        #     optimizer.step()
        #     losses.append(loss.item())
        #     sys.stdout.write(
        #         "[%-60s] %d%%"
        #         % (
        #             "=" * (60 * i // len(data_dict["trainloader"])),
        #             (100 * i // len(data_dict["trainloader"])),
        #         )
        #     )
        #     sys.stdout.flush()
        #     sys.stdout.write(f", training sample {(i+1):.2f}")
        #     sys.stdout.flush()
        score_window.append(loss.item())
        scores.append(np.mean(score_window))
        # net.eval()
        # val_losses = []
        # for i, data in enumerate(data_dict["valloader"], 1):
        #     sys.stdout.write("\r")
        #     inputs, targets = data.values()
        #     outputs = net(inputs)
        #     val_loss = criterion(outputs, targets.squeeze(-1)).sum()
        #     val_losses.append(val_loss.item())
        #     sys.stdout.write(
        #         "[%-60s] %d%%"
        #         % (
        #             "=" * (60 * i // len(data_dict["valloader"])),
        #             (100 * i // len(data_dict["valloader"])),
        #         )
        #     )
        #     sys.stdout.flush()
        #     sys.stdout.write(f", validation sample {(i+1):.2f}")
        #     sys.stdout.flush()
        #     if i == 10:
        #         break
        # val_window.append(sum(val_losses))
        # val_scores.append(np.mean(val_window))
        # net.train()
        sys.stdout.flush()
        sys.stdout.write(f", epoch {epoch}")
        sys.stdout.flush()
        sys.stdout.write(f", loss {np.mean(score_window):.4f}")
        sys.stdout.flush()
        # print(
        #     f"\nTraining loss {np.mean(score_window):.4f}, Val loss {np.mean(val_window):.4f}, Epoch {epoch}"
        # )
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
        default=DataTypes.CAPITALS,
        type=str,
        metavar=f"[{DataTypes.LETTERS},{DataTypes.CAPITALS}]",
        help="Which dataset to train on",
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of training epochs", default=1, type=int
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
    trainloader = return_dataloader(
        dataset["trainX"], dataset["trainY"], category="classification"
    )
    valloader = return_dataloader(
        dataset["valX"], dataset["valY"], category="classification"
    )
    data_dict = {"trainloader": trainloader, "valloader": valloader}
    train(data_dict, agent_params, training_params, dataset)
