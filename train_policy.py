from collections import deque
import sys
import torch
from torch import optim
import numpy as np
from globals import index_result_dict, PolicyOutputs
from experiments.generate_data import load_data
from experiments.globals import actionSpace, DataTypes, NetworkConfig, dataMapping
from torch.nn import SmoothL1Loss


def test_policy(agent_params, training_params, network_params, dataset):
    target_actions, target_rewards = dataset["trainY"]
    # print("target_actions", target_actions)
    # print("target_rewards", target_rewards)
    net = agent_params["network"](network_params)
    value_criterion = SmoothL1Loss()
    if training_params["resume"]:
        net.load_state_dict(torch.load(training_params["load_path"]))
    optimizer = optim.Adam(net.parameters(), lr=agent_params["learning_rate"])
    criterion = training_params["criterion"](reduction="sum")
    scores = []
    score_window = deque(maxlen=100)
    for epoch in range(training_params["epochs"]):
        sys.stdout.write("\r")
        optimizer.zero_grad()
        states = dataset["trainX"]
        outputs: PolicyOutputs = net(torch.as_tensor(states).long())
        target_actions, target_rewards = dataset["trainY"]
        # print("outputs.probs", outputs.probs)
        # print("outputs.value", outputs.value)
        policy_loss = criterion(outputs.probs, torch.as_tensor(target_actions))
        value_loss = value_criterion(
            outputs.value, torch.as_tensor(target_rewards).float()
        )
        loss = policy_loss + value_loss
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
    validation(net, dataset)
    # return np.mean(score_window)


def validation(network, dataset):
    while True:
        states = dataset["trainX"]
        target_actions, rewards = dataset["trainY"]
        print(f"Number of states {len(states)}")
        try:
            sample_idx = int(input(f"Pick a number between 0-{len(states)}"))
        except Exception as e:
            print(e)
        # sample_idx = np.random.choice(len(states))
        state = states[sample_idx][None, :]
        target_action = target_actions[sample_idx]
        target_reward = rewards[sample_idx]

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
        "-d",
        "--datatype",
        default=DataTypes.POLICY,
        type=str,
        metavar=f"[{DataTypes.THRESHOLD},{DataTypes.WORDLE},{DataTypes.RANDOM},{DataTypes.POLICY}]",
        help="Which dataset to train on",
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of training epochs", default=100, type=int
    )
    parser.add_argument(
        "--resume", help="resume training from an earlier run", action="store_true"
    )
    parser.add_argument(
        "-lr", help="resume training from an earlier run", type=float, default=0.03
    )
    parser.add_argument(
        "-v", dest="validate", help="validate the trained network", action="store_true"
    )
    parser.set_defaults(resume=False)
    parser.set_defaults(validate=False)

    args = parser.parse_args()
    print(args)
    network_path = "weights/policy_test"
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
    if args.validate:
        net = agent_params["network"](network_params)
        net.load_state_dict(torch.load(network_path))
        net.eval()
        validation(net, dataset)
    else:
        states = np.load("states.npy")
        actions = np.load("actions.npy")
        rewards = np.load("reward_targets.npy")[:, None]
        print(states.shape)
        print(actions.shape)
        print(rewards.shape)
        print(sum(rewards))
        dataset = {"trainX": states, "trainY": (actions, rewards)}
        test_policy(agent_params, training_params, network_params, dataset)
