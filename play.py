from collections import deque
import copy
import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch import optim
import numpy as np
from wordle import Wordle
from ML.networks import MuZeroNet
from MCTS_mu import MCTS, Node
from config import Config
from globals import (
    DynamicOutputs,
    Embeddings,
    PolicyOutputs,
    dictionary_index_to_word,
    dictionary_word_to_index,
    result_index_dict,
    index_result_dict,
)
import os
import ray


def create_root(state, reward):
    root = Node(None, 1)
    root.state = torch.as_tensor(state).long().unsqueeze(0)
    root.reward = reward
    return root


# make network turn work for batches
# implement learning update
# test outcome


def learning_update(
    mu_zero,
    params,
    states,
    actions,
    results,
    rewards,
) -> tuple((float, float)):
    # dynamics update
    dynamics_optimizer = params["dynamics_optimizer"]
    policy_optimizer = params["policy_optimizer"]
    dynamics_optimizer.zero_grad()
    dynamics_outputs: DynamicOutputs = mu_zero.dynamics(
        states,
        actions,
    )
    dynamic_loss = F.cross_entropy(
        dynamics_outputs.state_probs, results, reduction="sum"
    )
    dynamic_loss.backward()
    dynamics_optimizer.step()

    # policy update
    policy_outputs: PolicyOutputs = mu_zero.policy(states)
    policy_optimizer.zero_grad()
    policy_loss = F.cross_entropy(policy_outputs.probs, actions, reduction="sum")
    value_loss = F.smooth_l1_loss(
        rewards.unsqueeze(1), policy_outputs.value, reduction="sum"
    )
    policy_loss = policy_loss + value_loss  #
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(mu_zero._policy.parameters(), config.gradient_clip)
    policy_optimizer.step()
    return (policy_loss.item(), dynamic_loss.item())


def prepare_inputs(
    states,
    actions,
    results,
    reward_targets,
):
    states = np.array(states)
    actions = np.array(actions)
    reward_targets = np.array(reward_targets)
    result_targets = [torch.as_tensor(result_index_dict[tuple(res)]) for res in results]

    assert (
        len(states) == len(actions) == len(result_targets) == len(reward_targets)
    ), f"Improper lengths {print(len(states), len(actions), len(result_targets), len(reward_targets))}"
    # np.save("states.npy", states)
    # np.save("actions.npy", actions)
    # np.save("reward_targets.npy", reward_targets)
    states = torch.as_tensor(states).long()
    actions = torch.as_tensor(actions).long()
    result_targets = torch.stack(result_targets).long()
    reward_targets = torch.as_tensor(reward_targets).float()
    return states, actions, result_targets, reward_targets


def test_mcts_training(env, mcts: MCTS, mu_agent, config, params, training_params):
    if not training_params["skip_training"]:
        loss_window = deque(maxlen=100)
        policy_loss_window = deque(maxlen=100)
        dynamic_loss_window = deque(maxlen=100)
        for e in range(training_params["epochs"]):
            states = []
            actions = []
            results = []
            reward_targets = []
            word_targets = []
            for _ in range(training_params["trajectories"]):
                state, reward, done = env.reset()
                states.append(copy.deepcopy(state))
                word_targets.append(dictionary_word_to_index[env.word])
                # print("turn", env.turn)
                # print(env.word)
                while not done:
                    root = create_root(torch.as_tensor(state).long(), reward)
                    mcts.run(root, mu_agent)
                    # get chosen action
                    action, _ = root.select_child(config)
                    actions.append(action)
                    state, reward, done = env.step(dictionary_index_to_word[action])
                    results.append(state[env.turn - 1, :, Embeddings.RESULT])
                    if not done:
                        states.append(copy.deepcopy(state))
                        word_targets.append(dictionary_word_to_index[env.word])

                reward_targets.extend(
                    [reward * config.discount_rate ** i for i in range(env.turn)][::-1]
                )
                # shape inputs
            states, actions, result_targets, reward_targets = prepare_inputs(
                states, actions, results, reward_targets
            )
            for _ in range(1):
                sys.stdout.write("\r")
                losses = learning_update(
                    mu_agent,
                    params,
                    states,
                    actions,
                    result_targets,
                    reward_targets,
                )
                # print(losses)
                policy_loss_window.append(losses[0])
                dynamic_loss_window.append(losses[1])
                loss_window.append(sum(losses))
                sys.stdout.write(
                    "[%-60s] %d%%"
                    % (
                        "=" * (60 * (e + 1) // training_params["epochs"]),
                        (100 * (e + 1) // training_params["epochs"]),
                    )
                )
                sys.stdout.flush()
                sys.stdout.write(f" loss: {round(np.mean(loss_window),3)},")
                sys.stdout.flush()
                sys.stdout.write(
                    f" policy_loss: {round(np.mean(policy_loss_window),2)},"
                )
                sys.stdout.flush()
                sys.stdout.write(
                    f" dynamic_loss: {round(np.mean(dynamic_loss_window),2)},"
                )
                sys.stdout.flush()
            # lr_stepper.step()
        print(f"\nSaving weights to {training_params['load_path']}")
        torch.save(mu_agent.state_dict(), training_params["load_path"])
    validation(mu_agent, states, actions, result_targets, reward_targets)


def validation(network, states, target_actions, results, reward_targets):
    while True:
        print(f"Number of states {len(states)}")
        try:
            sample_idx = int(input(f"Pick a number between 0-{len(states)}"))
        except Exception as e:
            print(e)
        # sample_idx = np.random.choice(len(states))
        state = states[sample_idx][None, :]
        target_action = target_actions[sample_idx]
        target_result = index_result_dict[results[sample_idx].item()]
        reward_target = reward_targets[sample_idx]

        state = torch.as_tensor(state).long()
        print("state", state, state.shape)
        with torch.no_grad():
            policy_outputs: PolicyOutputs = network.policy(state)
            dynamic_outputs: DynamicOutputs = network.dynamics(
                state, policy_outputs.action
            )
            print("\naction", policy_outputs.action)
            print("target_action", target_action)
            print("\nvalue", policy_outputs.value)
            print("reward_target", reward_target)
            print("\nresult", dynamic_outputs.next_state)
            print("target_result", target_result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""
        Train wordle bot
        """
    )

    parser.add_argument(
        "--resume",
        "-r",
        dest="resume",
        action="store_true",
        help="resume training",
    )
    parser.add_argument(
        "--skip_training",
        dest="skip_training",
        action="store_true",
        help="skip training and validate only",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        dest="epochs",
        default=50,
        type=int,
        help="training epochs",
    )
    parser.add_argument(
        "--trajectories",
        "-t",
        dest="trajectories",
        default=25,
        type=int,
        help="number of trajectory samples to get",
    )
    args = parser.parse_args()
    print("Number of processors: ", mp.cpu_count())
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    ray.init()

    config = Config()
    env = Wordle(word_restriction=5)
    mu_agent = MuZeroNet({"nA": 5})
    mcts = MCTS(config)
    training_params = {
        "epochs": args.epochs,
        "trajectories": 25,
        "load_path": "weights/muzero",
        "resume": args.resume,
        "skip_training": args.skip_training,
        "learning_rate": 1e-3,
    }
    params = {
        "dynamics_optimizer": optim.Adam(
            mu_agent._dynamics.parameters(),
            lr=training_params["learning_rate"],
        ),
        "policy_optimizer": optim.Adam(
            mu_agent._policy.parameters(),
            lr=training_params["learning_rate"],
            weight_decay=config.L2,
        ),
    }
    # lr_stepsize = training_params["epochs"] // 5
    # lr_stepper = StepLR()
    # lr_stepper = MultiStepLR(
    #     optimizer=optimizer,
    #     milestones=[lr_stepsize * 2, lr_stepsize * 3, lr_stepsize * 4],
    #     gamma=0.1,
    # )
    if training_params["resume"]:
        print(f'loading network from {training_params["load_path"]}')
        mu_agent.load_state_dict(torch.load(training_params["load_path"]))
    # test_mcts_training(env, mcts, mu_agent, config, params, training_params)
    ray.shutdown()

# test_state4 = torch.tensor(
#     [
#         [[1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 1.0]],
#         [[1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 1.0]],
#         [[1.0, 3.0], [1.0, 3.0], [18.0, 1.0], [7.0, 1.0], [8.0, 2.0]],
#         [[1.0, 3.0], [1.0, 3.0], [18.0, 1.0], [7.0, 1.0], [8.0, 2.0]],
#         [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
#         [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
#     ],
# ).unsqueeze(0)

# test_state5 = torch.tensor(
#     [
#         [[1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 1.0]],
#         [[1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 1.0]],
#         [[1.0, 3.0], [1.0, 3.0], [18.0, 1.0], [7.0, 1.0], [8.0, 2.0]],
#         [[1.0, 3.0], [1.0, 3.0], [18.0, 1.0], [7.0, 1.0], [8.0, 2.0]],
#         [[1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 1.0]],
#         [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
#     ],
# ).unsqueeze(0)

# test_state6 = torch.tensor(
#     [
#         [[1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 1.0]],
#         [[1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 1.0]],
#         [[1.0, 3.0], [1.0, 3.0], [18.0, 1.0], [7.0, 1.0], [8.0, 2.0]],
#         [[1.0, 3.0], [1.0, 3.0], [18.0, 1.0], [7.0, 1.0], [8.0, 2.0]],
#         [[1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 1.0]],
#         [[1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 1.0]],
#     ]
# ).unsqueeze(0)

# states = torch.cat((test_state6, test_state5, test_state4))
# turns = torch.count_nonzero(states, dim=1)[:, 0, 0].view(-1, 1)
# bools = torch.where(turns >= 5, 1, 0)
# print(bools)
# # turns = extract_turn(states)
# print(turns)

# B = states.shape[0]
# bools = []
# for i in range(B):
#     mask = torch.where(test_state5 == 0)[1]
#     if mask:
#         bools.append(mask[0])
#     else:
#         bools.append(1)
# bools = [torch.where(states[i, :, :, :] == 0)[1] for i in range(B)]
# # print(states.shape)
# # mask = torch.where(states[:, :, 1] > 0)
# print(bools)
# print(torch.where(test_state5 == 0)[1])
# print(torch.where(test_state6 == 0)[1])
# t5 = torch.where(test_state5 == 0)[1][0].unsqueeze(0)
# t6 = torch.where(test_state5 == 0)[1][0].unsqueeze(0)
# print(t5)
# print(t6)
# test_mcts_training(env, mcts, mu_agent, config)
# a = 11
# b = -2
# maxlen = max(len(bin(a)), len(bin(b)))
# # a = a.zfill(maxlen)
# # b = b.zfill(maxlen)
# print(bin(11))
# print(bin(4))
# print(11 ^ 4, bin(11 ^ 4))
# print(bin(a))
# print(bin(b))
# print(a ^ b, bin(a ^ b))
# print(a & b, bin(a & b))
# print((a & b) << 1)

# a = torch.ones((19, 6, 5, 8))
# emb = nn.Embedding(2, 8)
# x = emb(a.long())
# print(x.shape)

# a = torch.tensor(
#     [
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [1, 0],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#     ]
# )

# res = torch.where(a == 1)[1]
# print(res)
