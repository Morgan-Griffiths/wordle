from collections import deque
import copy
import sys
import ray
import torch
from torch import optim
import numpy as np
from ML.networks import StateActionTransition
from config import Config
from globals import DynamicOutputs, Mappings, index_result_dict, CHECKPOINT
from experiments.globals import (
    LearningCategories,
    NetworkConfig,
)
import torch.nn.functional as F
from main import load_replay_buffer
from ray_files.replay_buffer import ReplayBuffer
from wordle import Wordle


def test_state_transition(net, training_params, agent_params, per_buffer):
    optimizer = optim.AdamW(
        net.parameters(), lr=agent_params["learning_rate"], weight_decay=3e-2
    )
    scores = []
    score_window = deque(maxlen=100)
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
    # print(state_batch)
    # print(action_batch)
    for epoch in range(training_params["epochs"]):
        sys.stdout.write("\r")
        outputs: DynamicOutputs = net.dynamics(
            state_batch,
            action_batch.unsqueeze(1),
        )
        loss = F.nll_loss(outputs.state_logprobs, result_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        score_window.append(loss.item())
        scores.append(np.mean(score_window))
        sys.stdout.flush()
        sys.stdout.write(f", epoch {epoch}")
        sys.stdout.flush()
        sys.stdout.write(f", loss {np.mean(score_window):.4f}")
        sys.stdout.flush()
    # print(f"Saving weights to {training_params['load_path']}")
    # torch.save(net.state_dict(), training_params["load_path"])
    validation(net, batch)


def validation(network, batch):
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
            sample_idx = int(input(f"Pick a number between 0-{len(state_batch)-1}"))
        except Exception as e:
            print(e)
        # sample_idx = np.random.choice(len(states))
        state, action = state_batch[sample_idx], action_batch[sample_idx]
        target_result = result_batch[sample_idx]
        with torch.no_grad():
            outputs: DynamicOutputs = network.dynamics(
                state.unsqueeze(0), action.view(1, 1)
            )
            print("state", state)
            print("action", action)
            print(
                "actual state prob",
                outputs.state_probs[0][target_result.item()],
            )
            print("winning state prob", outputs.state_probs[0][-1])
            print("target_result", index_result_dict[target_result.item()])


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
    parser.add_argument("-b", "--batch", help="Batch size", default=4, type=int)
    parser.add_argument(
        "--resume", help="resume training from an earlier run", action="store_true"
    )
    parser.add_argument(
        "-lr", help="resume training from an earlier run", type=float, default=3e-3
    )
    parser.add_argument(
        "-v", dest="validate", help="validate the trained network", action="store_true"
    )
    parser.set_defaults(resume=False)
    parser.set_defaults(validate=False)
    args = parser.parse_args()
    print(args)

    config = Config()
    config.lr_init = args.lr
    config.batch_size = args.batch
    buffer_info = load_replay_buffer()
    checkpoint = copy.copy(CHECKPOINT)
    checkpoint["num_played_steps"] = buffer_info["num_played_steps"]
    checkpoint["num_played_games"] = buffer_info["num_played_games"]
    checkpoint["num_reanalysed_games"] = buffer_info["num_reanalysed_games"]
    per_buffer = ReplayBuffer.remote(checkpoint, buffer_info["buffer"], config)
    # mu_zero = MuZeroNet(config)

    mappings = Mappings(config.word_restriction)
    env = Wordle(mappings)
    mu_zero = StateActionTransition(config)

    network_path = "weights/dynamics"
    # loss_type = dataMapping[args.datatype]
    agent_params = {
        "learning_rate": args.lr,
        "save_dir": "checkpoints",
        "save_path": network_path,
        "load_path": network_path,
    }
    training_params = {
        "resume": args.resume,
        "epochs": args.epochs,
        "criterion": NetworkConfig.LossFunctions[
            LearningCategories.MULTICLASS_CATEGORIZATION
        ],
        "load_path": network_path,
    }
    # network_params = {
    #     "seed": 346,
    #     "nA": actionSpace[args.datatype],
    #     "load_path": network_path,
    #     "emb_size": 16,
    # }
    # dataset = load_data(args.datatype)
    # if args.validate:
    #     net = agent_params["network"](network_params)
    #     net.load_state_dict(torch.load(network_path))
    #     net.eval()
    #     validation(net, dataset)
    # else:
    test_state_transition(mu_zero, training_params, agent_params, per_buffer)
