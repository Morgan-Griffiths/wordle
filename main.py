import copy
import pathlib
import pickle
import ray
from ML.muzero import MuZero
from ML.utils import load_replay_buffer, load_model_menu

from ray_files.utils import CPUActor
from ray_files.validate_model import ValidateModel
from ray_files.trainer import Trainer
from ray_files.shared_storage import SharedStorage
from ray_files.replay_buffer import ReplayBuffer

from config import Config
from wordle import Wordle
from globals import CHECKPOINT, WordDictionaries
from create_dynamic_dataset import create_dataset


def play_wordle():
    env = Wordle()
    print(env.word)
    for i in range(6):
        while True:
            word = input("input a 5 letter word ").strip()
            contains_digit = False
            for letter in word:
                if letter.isdigit():
                    contains_digit = True
            if len(word) != 5 or contains_digit:
                continue
            else:
                break
        result, reward, done = env.step(word)
        print(result, reward, done)


def model_update_step(model, buffer_info):
    config = Config()
    config.train_on_gpu = False
    word_dictionary = WordDictionaries(config.word_restriction)
    checkpoint = copy.copy(CHECKPOINT)
    checkpoint["num_played_steps"] = buffer_info["num_played_steps"]
    checkpoint["num_played_games"] = buffer_info["num_played_games"]
    checkpoint["num_reanalysed_games"] = buffer_info["num_reanalysed_games"]
    cpu_actor = CPUActor.remote()
    cpu_weights = cpu_actor.get_initial_weights.remote(config, word_dictionary)
    checkpoint["weights"], _ = copy.deepcopy(ray.get(cpu_weights))
    per_buffer = ReplayBuffer.remote(
        checkpoint, buffer_info["buffer"], config, word_dictionary
    )
    trainer = Trainer.options(num_cpus=0, num_gpus=0,).remote(
        checkpoint,
        config,
        word_dictionary,
    )
    shared_storage_worker = SharedStorage.remote(
        checkpoint,
        config,
    )
    vm = ValidateModel(checkpoint, config, word_dictionary)
    vm.check_model_updates(trainer, per_buffer, shared_storage_worker)


def validate_buffer(buffer_info):
    config = Config()
    config.train_on_gpu = False
    config.batch_size = 4
    word_dictionary = WordDictionaries(config.word_restriction)
    checkpoint = copy.copy(CHECKPOINT)
    checkpoint["num_played_steps"] = buffer_info["num_played_steps"]
    checkpoint["num_played_games"] = buffer_info["num_played_games"]
    checkpoint["num_reanalysed_games"] = buffer_info["num_reanalysed_games"]
    per_buffer = ReplayBuffer.remote(
        checkpoint, buffer_info["buffer"], config, word_dictionary
    )
    try:
        while True:
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
            print(f"\nindex_batch {index_batch}")
            print(f"\nstate_batch {state_batch}")
            print(f"\naction_batch {action_batch}")
            print(f"\nvalue_batch {value_batch}")
            print(f"\nreward_batch {reward_batch}")
            print(f"\npolicy_batch {policy_batch}")
            print(f"\nweight_batch {weight_batch}")
            print(f"\nresult_batch {result_batch}")
            print(f"\ngradient_scale_batch {gradient_scale_batch}")
            input(f"press ENTER to continue, or ctrl c to quit")
    except KeyboardInterrupt:
        return


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
        "--epochs",
        "-e",
        dest="epochs",
        default=5,
        type=int,
        help="training epochs",
    )
    parser.add_argument(
        "--sims",
        "-s",
        dest="sims",
        default=50,
        type=int,
        help="number of times to search the MCTS tree",
    )
    parser.add_argument(
        "--no_gpu",
        dest="no_gpu",
        default=False,
        action="store_true",
        help="training epochs",
    )
    args = parser.parse_args()

    print("args", args)
    # main(args.resume, args.model, args.epochs, args.train_type)
    config = Config()
    config.load_dynamic_weights = False
    config.train_on_gpu = not args.no_gpu
    config.training_steps = args.epochs
    config.num_simulations = args.sims
    with MuZero(config) as mu_zero:
        try:
            while True:
                # Configure running options
                options = [
                    "Train",
                    "Load pretrained model",
                    "Validate Model",
                    "Validate MCTS",
                    "Load and examine buffer",
                    "check model updates",
                    "Exit",
                ]
                print()
                for i in range(len(options)):
                    print(f"{i}. {options[i]}")

                choice = input("Enter a number to choose an action: ")
                valid_inputs = [str(i) for i in range(len(options))]
                while choice not in valid_inputs:
                    choice = input("Invalid input, enter a number listed above: ")
                choice = int(choice)
                if choice == 0:
                    mu_zero.train()
                elif choice == 1:
                    load_model_menu(mu_zero)
                elif choice == 2:
                    mu_zero.validate()
                elif choice == 3:
                    mu_zero.validate_mcts()
                elif choice == 4:
                    buffer_info = load_replay_buffer()
                    validate_buffer(buffer_info)
                elif choice == 5:
                    buffer_info = load_replay_buffer()
                    model_update_step(mu_zero, buffer_info)
                else:
                    break
        except KeyboardInterrupt:
            ...
