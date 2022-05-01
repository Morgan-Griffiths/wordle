from collections import deque
import sys
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from globals import index_result_dict, PolicyOutputs,CHECKPOINT
from ray_files.replay_buffer import ReplayBuffer
from experiments.generate_data import load_data
from experiments.globals import actionSpace, DataTypes, NetworkConfig, dataMapping
from torch.nn import SmoothL1Loss
from config import Config
from main import load_replay_buffer
import copy
import ray

def test_policy(agent_params, training_params, network_params, per_buffer):
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
        outputs: PolicyOutputs = net(state_batch)
        # print("outputs.probs", outputs.probs)
        # print("outputs.value", outputs.value)
        policy_loss = F.nll_loss(outputs.logprobs, word_batch)
        value_loss = value_criterion(
            outputs.value, reward_batch
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
        try:
            sample_idx = int(input(f"Pick a number between 0-{len(state_batch)}"))
        except Exception as e:
            print(e)
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
    config = Config()
    print(args)
    network_path = "results/policy_test"
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
    config.batch_size = 4
    buffer_info = load_replay_buffer()
    checkpoint = copy.copy(CHECKPOINT)
    checkpoint["num_played_steps"] = buffer_info["num_played_steps"]
    checkpoint["num_played_games"] = buffer_info["num_played_games"]
    checkpoint["num_reanalysed_games"] = buffer_info["num_reanalysed_games"]
    per_buffer = ReplayBuffer.remote(checkpoint, buffer_info["buffer"], config)
    if args.validate:
        net = agent_params["network"](network_params)
        net.load_state_dict(torch.load(network_path))
        net.eval()
        validation(net, per_buffer)
    else:
        test_policy(agent_params, training_params, network_params, per_buffer)
