from collections import deque
import copy
import sys
import ray
from ray.train import Trainer
import torch
from torch import optim
import numpy as np
from MCTS_mu import GameHistory
import torch.multiprocessing as mp
from ML.networks import MuZeroNet, TestNet, StateActionTransition
from config import Config
from globals import (
    DynamicOutputs,
    Embeddings,
    PolicyOutputs,
    index_result_dict,
    CHECKPOINT,
)
from experiments.generate_data import load_data
from experiments.globals import (
    LearningCategories,
    actionSpace,
    DataTypes,
    NetworkConfig,
    dataMapping,
)
from collections import Counter
import torch.nn.functional as F
from ray_files.replay_buffer import ReplayBuffer
from wordle import Wordle


def gather_trajectories(env, per_buffer, config):
    for _ in range(config.num_warmup_games):
        game_history = GameHistory()
        state, reward, done = env.reset()
        game_history.state_history.append(state.copy())
        game_history.word_history.append(env.word_to_action(env.word))
        while not done:

            action = np.random.randint(0, config.action_space)
            state, reward, done = env.step(env.action_to_string(action))
            # Next batch
            game_history.result_history.append(
                state[env.turn - 1, :, Embeddings.RESULT]
            )
            game_history.action_history.append(action)
            game_history.reward_history.append(reward)
            if not done:
                game_history.state_history.append(state.copy())
                game_history.word_history.append(env.word_to_action(env.word))
        for i in range(env.turn):
            # if i % 2 == 0:
            game_history.child_visits.append([a for a in range(config.action_space)])
            game_history.root_values.append(1)
            game_history.max_actions.append(0)
            # else:
            #     game_history.child_visits.append(np.arange(config.action_space))
        # game_history.child_visits = np.array(game_history.child_visits)
        per_buffer.save_game.remote(game_history)
    return per_buffer


def train_dynamics(config, training_params, per_buffer):
    # device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    # print('device',device)
    # net = net.to(device)
    net = StateActionTransition(config)
    # net = ray.train.torch.prepare_model(net)
    optimizer = optim.AdamW(
        net.parameters(), lr=config.lr_init, weight_decay=3e-2
    )

    scores = []
    score_window = deque(maxlen=100)
    try:
        for _ in range(config.num_warmup_training_steps):
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
            # state_batch = state_batch.to(device)
            # action_batch = action_batch.to(device)
            # result_batch = result_batch.to(device)
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
            print(f"Saving weights to {training_params['load_path']}")
            torch.save(net.state_dict(), training_params["load_path"])
    
        validation(net, batch)
    except KeyboardInterrupt:
        pass 
    per_buffer = None


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
    parser.add_argument("-b", "--batch", help="Batch size", default=4096, type=int)
    parser.add_argument(
        "--resume", help="resume training from an earlier run", action="store_true"
    )
    parser.add_argument(
        "-lr", help="resume training from an earlier run", type=float, default=3e-3
    )
    parser.add_argument(
        "-v", dest="validate", help="validate the trained network", action="store_true"
    )
    parser.add_argument(
        "-g", dest="warmup_games", help="number of warmup games to play", default=50
    )
    parser.add_argument(
        "-s", dest="warmup_steps", help="number of dynamics training steps", default=10
    )
    parser.set_defaults(resume=False)
    parser.set_defaults(validate=False)
    args = parser.parse_args()
    print(args)

    config = Config()
    config.lr_init = args.lr
    config.batch_size = args.batch
    config.num_warmup_games = args.warmup_games
    config.num_warmup_training_steps = args.warmup_steps
    # buffer_info = load_replay_buffer()
    checkpoint = copy.copy(CHECKPOINT)
    # checkpoint["num_played_steps"] = buffer_info["num_played_steps"]
    # checkpoint["num_played_games"] = buffer_info["num_played_games"]
    # checkpoint["num_reanalysed_games"] = buffer_info["num_reanalysed_games"]
    per_buffer = ReplayBuffer.remote(checkpoint, {}, config)

    env = Wordle(word_restriction=config.action_space)
    config.word_to_index = env.dictionary_word_to_index
    config.index_to_word = env.dictionary_index_to_word

    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # if config.train_on_gpu:
    #     num_gpus = torch.cuda.device_count()
    # else:
    #     num_gpus = 0
    # print("Is gpu avail ", torch.cuda.is_available())
    # print("Number of processors: ", mp.cpu_count())
    # print(f"Number of GPUs: {num_gpus}")
    # total_gpus = num_gpus
    # ray.init(num_gpus=total_gpus, ignore_reinit_error=True)
    # mu_zero = MuZeroNet(config)
    # mu_zero = StateActionTransition(config)

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
    per_buffer = gather_trajectories(env, per_buffer, config)
    train_dynamics(config, training_params, per_buffer)

    # trainer = Trainer(backend="torch", num_workers=4, use_gpu=True)
    # trainer.start()
    # results = trainer.run(train_func_distributed,config=config,training_params=training_params,per_buffer=per_buffer)
    # trainer.shutdown()
