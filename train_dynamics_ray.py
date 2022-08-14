import copy
import math
import pathlib
import pickle
import time
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import ray
import numpy as np

from config import Config
from wordle import Wordle
from globals import CHECKPOINT, Mappings
from ray_files.utils import CPUActor
from ray_files.trainer import Trainer
from ray_files.shared_storage import SharedStorage
from ray_files.replay_buffer import ReplayBuffer, Reanalyse
from ray_files.self_play import SelfPlay


class MuDyno:
    def __init__(self, config):
        self.config = config
        self.mappings = Mappings(config.word_restriction)
        self.env = Wordle(self.mappings)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if config.train_on_gpu:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = 0
        print("Is gpu avail ", torch.cuda.is_available())
        print("Number of processors: ", mp.cpu_count())
        print(f"Number of GPUs: {self.num_gpus}")

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = copy.copy(CHECKPOINT)
        self.replay_buffer = {}

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        """
        Spawn ray workers and launch the training.
        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        # Manage GPUs
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + log_in_tensorboard * self.config.selfplay_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
            num_gpus_per_worker -= 0.05
        else:
            num_gpus_per_worker = 0
        print("num_gpus_per_worker", num_gpus_per_worker)
        # Initialize workers
        self.training_worker = Trainer.options(
            num_cpus=0,
            num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(self.checkpoint, self.config, self.mappings)
        self.shared_storage_worker = SharedStorage.options(
            num_cpus=0,
            num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
        ).remote(
            self.checkpoint,
            self.config,
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = ReplayBuffer.remote(
            self.checkpoint,
            self.replay_buffer,
            self.config,
        )

        if self.config.use_last_model_value:
            self.reanalyse_worker = Reanalyse.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            ).remote(self.checkpoint, self.config, self.mappings)

        self.self_play_workers = [
            SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            ).remote(
                self.checkpoint,
                self.env,
                self.config,
                self.mappings,
                self.config.seed + seed,
            )
            for seed in range(self.config.num_workers)
        ]

        # Launch workers
        [
            self_play_worker.gather_trajectories.remote(
                self.env.copy(), self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_play_worker in self.self_play_workers
        ]
        self.training_worker.continuous_dynamic_weight_updates.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )
        if log_in_tensorboard:
            self.logging_loop(
                num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            )

    def logging_loop(self, num_gpus):
        """
        Keep track of the training performance.
        """
        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation
        writer.add_text(
            "Model summary",
            self.summary,
        )

        # Loop for updating the training performance
        counter = 0
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "actor_loss",
            "dynamic_loss",
            "value_loss",
            "policy_loss",
            "actor_probs",
            "actor_value",
            "dynamic_prob_winning_state",
            "results",
            "actions",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["training_step"] < self.config.num_warmup_training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_histogram(
                    "2.Dynamics/dynamic_prob_winning_state",
                    info["dynamic_prob_winning_state"],
                    counter,
                )
                writer.add_histogram(
                    "2.Dynamics/results",
                    info["results"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/1.Self_played_games",
                    info["num_played_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Training_steps", info["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/4.Reanalysed_games",
                    info["num_reanalysed_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/5.Training_steps_per_self_played_step_ratio",
                    info["training_step"] / max(1, info["num_played_steps"]),
                    counter,
                )
                writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
                )
                writer.add_scalar("3.Loss/dynamic_loss", info["dynamic_loss"], counter)
                writer.add_scalar("3.Loss/total_loss", info["total_loss"], counter)
                print(
                    f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.num_warmup_training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                    end="\r",
                )
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        self.terminate_workers(writer)

    def terminate_workers(self, writer):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        writer.close()

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.
        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.
            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            parent_dir = pathlib.Path(__file__).resolve().parents[0]
            checkpoint_path = pathlib.Path(parent_dir / checkpoint_path)
            self.checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_played_steps"] = replay_buffer_infos[
                "num_played_steps"
            ]
            self.checkpoint["num_played_games"] = replay_buffer_infos[
                "num_played_games"
            ]
            self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                "num_reanalysed_games"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print(f"Using empty buffer.")
            self.replay_buffer = {}
            self.checkpoint["training_step"] = 0
            self.checkpoint["num_played_steps"] = 0
            self.checkpoint["num_played_games"] = 0
            self.checkpoint["num_reanalysed_games"] = 0

    def __enter__(self):
        ray.init(num_gpus=self.num_gpus, ignore_reinit_error=True)
        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config, self.mappings)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

    def __exit__(self, exc_type, exc_value, traceback):
        ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""
        Train and evaluate networks on letter representations.
        """
    )
    parser.add_argument("-b", "--batch", help="Batch size", default=128, type=int)
    parser.add_argument(
        "--resume", help="resume training from an earlier run", action="store_true"
    )
    parser.add_argument("-lr", help="learning rate", type=float, default=3e-3)
    parser.add_argument(
        "-v", dest="validate", help="validate the trained network", action="store_true"
    )
    parser.add_argument(
        "-g",
        "--games",
        dest="warmup_games",
        help="number of warmup games to play per thread",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--steps",
        dest="warmup_steps",
        help="number of dynamics training steps",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--no-gpu",
        dest="no_gpu",
        default=False,
        action="store_true",
        help="training epochs",
    )
    parser.set_defaults(resume=False)
    parser.set_defaults(validate=False)

    args = parser.parse_args()

    print("args", args)
    config = Config()
    config.PER = False
    config.train_on_gpu = not args.no_gpu
    config.num_warmup_training_steps = args.warmup_steps
    config.num_warmup_games = args.warmup_games
    config.load_dynamic_weights = False
    mu_dyno = MuDyno(config)

    with mu_dyno:
        mu_dyno.train()
