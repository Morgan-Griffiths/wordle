from calendar import day_name
from collections import deque
import sys
import torch
from torch import optim
import numpy as np
from globals import DynamicOutputs, PolicyOutputs, index_result_dict
from experiments.generate_data import load_data
from experiments.globals import actionSpace, DataTypes, NetworkConfig, dataMapping


def test_state_transition(agent_params, training_params, network_params, dataset):
    net = agent_params["network"](network_params)
    if training_params["resume"]:
        net.load_state_dict(torch.load(training_params["load_path"]))
    optimizer = optim.Adam(net.parameters(), lr=agent_params["learning_rate"])
    criterion = training_params["criterion"](reduction="sum")
    scores = []
    score_window = deque(maxlen=100)
    for epoch in range(training_params["epochs"]):
        sys.stdout.write("\r")
        optimizer.zero_grad()
        states, actions = dataset["trainX"]
        outputs: DynamicOutputs = net(
            torch.as_tensor(states).long(),
            torch.as_tensor(actions).long(),
        )
        target_results, target_rewards = dataset["trainY"]
        loss = criterion(outputs.state_probs, torch.as_tensor(target_results))
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
        states, actions = dataset["trainX"]
        target_results, target_rewards = dataset["trainY"]
        print(f"Number of states {len(states)}")
        try:
            sample_idx = int(input(f"Pick a number between 0-{len(states)}"))
        except Exception as e:
            print(e)
        # sample_idx = np.random.choice(len(states))
        state, action = states[sample_idx], actions[sample_idx]
        target_result, target_reward = (
            target_results[sample_idx],
            target_rewards[sample_idx],
        )
        state = torch.as_tensor(state).long().unsqueeze(0)
        action = torch.as_tensor(action).long().unsqueeze(0)
        print("state", state, state.shape)
        print("action", action, action.shape)
        with torch.no_grad():
            s_prime, s_logits, reward, reward_logits = network(state, action)
            print("s_prime", s_prime)
            print("target_result", index_result_dict[target_result])
            print("reward", reward)
            print("target_reward", target_reward)


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
        default=DataTypes.RANDOM,
        type=str,
        metavar=f"[{DataTypes.LETTERS},{DataTypes.THRESHOLD},{DataTypes.WORDLE},{DataTypes.MULTI_TARGET},{DataTypes.CONSTELLATION},{DataTypes.RANDOM}]",
        help="Which dataset to train on",
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of training epochs", default=100, type=int
    )
    parser.add_argument(
        "--resume", help="resume training from an earlier run", action="store_true"
    )
    parser.add_argument(
        "-lr", help="resume training from an earlier run", type=float, default=0.003
    )
    parser.add_argument(
        "-v", dest="validate", help="validate the trained network", action="store_true"
    )
    parser.set_defaults(resume=False)
    parser.set_defaults(validate=False)

    args = parser.parse_args()
    print(args)
    network_path = "weights/dynamics"
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
        test_state_transition(agent_params, training_params, network_params, dataset)
