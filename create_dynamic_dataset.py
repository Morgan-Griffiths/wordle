from wordle import Wordle
from globals import Embeddings, result_index_dict
import ray
from MCTS_mu import GameHistory
import tarfile
import numpy as np
import io
from config import Config
import time
from tqdm import tqdm
from ML.networks import MuZeroNet, StateActionTransition
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from ray import train
import ray.train.torch
from ray.train import Trainer, TrainingCallback
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import os
from prettytable import PrettyTable
from collections import deque
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import sys


class Storage:
    def __init__(self, config):
        # self.actions = zarr.zeros(shape=(config.num_warmup_games))
        # self.states = zarr.zeros(shape=(config.num_warmup_games,6,5,2))
        # self.labels = zarr.zeros(shape=(config.num_warmup_games))
        self.actions = np.zeros(shape=(config.num_dynamics_examples), dtype=np.uint16)
        self.states = np.zeros(
            shape=(config.num_dynamics_examples, 6, 5, 2), dtype=np.uint8
        )
        self.labels = np.zeros(shape=(config.num_dynamics_examples), dtype=np.uint8)
        self.n = 0
        self.max_n = config.num_dynamics_examples

    def save_game_history(self, states, actions, labels):
        for state, action, label in zip(states, actions, labels):
            if self.n >= self.max_n:
                break
            self.actions[self.n] = action
            self.labels[self.n] = result_index_dict[tuple(label)]
            self.states[self.n, :, :, :] = state
            self.n += 1
        return self.n >= self.max_n

    def save_state(self):
        np.save("data/actions", self.actions)
        np.save("data/labels", self.labels)
        np.save("data/states", self.states)

    def get_info(self):
        return self.states, self.actions, self.labels


def randomly_sample_games(shared_storage, config):
    env = Wordle(word_restriction=config.word_restriction)
    for _ in tqdm(range(config.num_dynamics_examples)):
        states = []
        actions = []
        labels = []
        state, reward, done = env.reset()
        states.append(state.copy())
        while not done:
            action = np.random.randint(1, config.action_space + 1)
            state, reward, done = env.step(env.action_to_string(action))
            # Next batch
            labels.append(state[env.turn - 1, :, Embeddings.RESULT])
            actions.append(action)
            if not done:
                states.append(state.copy())

        full = shared_storage.save_game_history(states, actions, labels)
        if full:
            break


def create_dataset():
    config = Config()
    config.num_dynamics_examples = 30000000
    config.num_workers = 2
    shared_storage = Storage(config)
    randomly_sample_games(shared_storage, config)
    shared_storage.save_state()


######################################################################3


def read_npy_chunk(filename, start_row, num_rows):
    """
    Reads a partial array (contiguous chunk along the first
    axis) from an NPY file.
    Parameters
    ----------
    filename : str
        Name/path of the file from which to read.
    start_row : int
        The first row of the chunk you wish to read. Must be
        less than the number of rows (elements along the first
        axis) in the file.
    num_rows : int
        The number of rows you wish to read. The total of
        `start_row + num_rows` must be less than the number of
        rows (elements along the first axis) in the file.
    Returns
    -------
    out : ndarray
        Array with `out.shape[0] == num_rows`, equivalent to
        `arr[start_row:start_row + num_rows]` if `arr` were
        the entire array (note that the entire array is never
        loaded into memory by this function).
    """
    assert start_row >= 0 and num_rows > 0
    with open(filename, "rb") as fhandle:
        major, minor = np.lib.format.read_magic(fhandle)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fhandle)
        assert not fortran, "Fortran order arrays not supported"
        # Make sure the offsets aren't invalid.
        assert start_row < shape[0], "start_row is beyond end of file"
        assert start_row + num_rows <= shape[0], "start_row + num_rows > shape[0]"
        # Get the number of elements in one 'row' by taking
        # a product over all other dimensions.
        row_size = np.prod(shape[1:])
        start_byte = start_row * row_size * dtype.itemsize
        fhandle.seek(int(start_byte), 1)
        n_items = int(row_size * num_rows)
        flat = np.fromfile(fhandle, count=n_items, dtype=dtype)
        return flat.reshape((-1,) + shape[1:])


class DynamicSamples(Dataset):
    def __init__(self):
        # self.labels = np.load('data/labels.npy')
        # self.actions = np.load('data/actions.npy')
        # self.states = np.load('data/states.npy')
        ...

    def __len__(self):
        return int(30e6)

    def __getitem__(self, index):
        label = torch.from_numpy(read_npy_chunk("data/labels.npy", index, 1))
        action = torch.from_numpy(
            read_npy_chunk("data/actions.npy", index, 1).astype(np.int16)
        )
        state = torch.from_numpy(read_npy_chunk("data/states.npy", index, 1))
        return (action, state, label)


class DynamicInMemory(Dataset):
    def __init__(self):
        self.labels = np.load("data/labels.npy")
        self.actions = np.load("data/actions.npy")
        self.states = np.load("data/states.npy")
        # self.labels = self.labels.astype(np.int8)
        self.actions = self.actions.astype(np.int16)
        # self.states = self.states.astype(np.int8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = torch.as_tensor(self.labels[index])
        action = torch.as_tensor(self.actions[index, None])
        state = torch.from_numpy(self.states[index])
        return (action, state, label)


def train_epoch(dataloader, model, loss_fn, optimizer):
    losses = []
    for action, state, label in dataloader:
        # state = state.squeeze(1).long().to(device)
        # label = label.squeeze(-1).to(device)
        # action = action.to(device)
        state = state.squeeze(1).long()
        label = label.squeeze(-1)
        print(action.shape, state.shape, label.shape)
        print(action.get_device())
        print(next(model.parameters()).device)
        pred = model.dynamics(state, action)
        loss = loss_fn(pred.state_logprobs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return {"loss": losses}


def train_func(config):
    training_data = DynamicInMemory()
    train_loader = DataLoader(training_data, batch_size=2048, shuffle=True)
    train_loader = train.torch.prepare_data_loader(train_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.train_on_gpu = False
    model = MuZeroNet(config)
    # model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
    # model.to(device)
    model = train.torch.prepare_model(model)
    # model.train()

    loss_fn = nn.NLLLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(config.beta_1, config.beta_2),
        lr=config.lr_init,
        weight_decay=config.weight_decay,
    )

    results = []

    for _ in range(config.num_warmup_training_steps):
        result = train_epoch(device, train_loader, model, loss_fn, optimizer)
        train.report(**result)
        results.append(result)

    return results


class PrintingCallback(TrainingCallback):
    def handle_result(self, results: list[dict], **info):
        print(results)


def train_linear(num_workers=2, use_gpu=False):
    trainer = Trainer(
        backend="torch",
        num_workers=num_workers,
        use_gpu=use_gpu,
        logdir="./ray_results",
    )
    config = Config()
    env = Wordle(word_restriction=config.word_restriction)
    config.word_to_index = env.dictionary_word_to_index
    config.index_to_word = env.dictionary_index_to_word
    trainer.start()
    results = trainer.run(
        train_func, config, callbacks=[PrintingCallback(), JsonLoggerCallback()]
    )
    trainer.shutdown()

    return results


def setup_world(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_network(id, data_dict, config, training_params):
    print(f"Process {id}")
    model = MuZeroNet(config)
    if training_params["resume"]:
        print(f"Loading weights from {training_params['load_path']}")
        load_weights(model, training_params["load_path"])
    if id == 0 or id == "cpu":
        count_parameters(model)
    if torch.cuda.device_count() > 1:
        setup_world(id, 2)
        model = DDP(model)
    else:
        id = "cpu"
    model.to(id)
    if "category_weights" in data_dict:
        print("using category weights")
        criterion = training_params["criterion"](
            data_dict["category_weights"].to(id), reduction="sum"
        )
    else:
        criterion = training_params["criterion"](reduction="sum")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(config.beta_1, config.beta_2),
        lr=config.lr_init,
        weight_decay=config.weight_decay,
    )
    lr_stepsize = training_params["epochs"] // 5
    lr_stepper = MultiStepLR(
        optimizer=optimizer,
        milestones=[lr_stepsize * 2, lr_stepsize * 3, lr_stepsize * 4],
        gamma=0.1,
    )
    scores = []
    score_window = deque(maxlen=100)
    for epoch in range(training_params["epochs"]):
        try:
            losses = []
            if id == 0 or id == "cpu":
                print(f"Saving weights to {training_params['load_path']}")
                torch.save(
                    model.state_dict(), training_params["load_path"] + f"{epoch}"
                )
            for i, (actions, states, labels) in enumerate(data_dict["trainloader"], 1):
                actions = actions.to(id)
                states = states.to(id).long()
                labels = labels.to(id)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(states, actions)
                loss = criterion(outputs.state_logprobs, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if id == 0:
                    sys.stdout.write("\r")
                    sys.stdout.write(
                        "[%-60s] %d%%"
                        % (
                            "=" * (60 * (i + 1) // len(data_dict["trainloader"])),
                            (100 * (i + 1) // len(data_dict["trainloader"])),
                        )
                    )
                    sys.stdout.flush()
                    sys.stdout.write(f", training sample {(i+1):.2f}")
                    sys.stdout.flush()
            lr_stepper.step()
            score_window.append(loss.item())
            scores.append(np.mean(score_window))
            if id == 0 or id == "cpu":
                print(f"\nTraining loss {np.mean(score_window):.4f}, Epoch {epoch}")
        except KeyboardInterrupt:
            break
    torch.save(model.state_dict(), training_params["load_path"] + f"{epoch}")
    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()


def train_multi():
    dataset = DynamicInMemory()
    trainloader = DataLoader(
        dataset=dataset, batch_size=4096, shuffle=True, num_workers=8
    )
    data_dict = {
        "trainloader": trainloader,
    }
    world_size = max(torch.cuda.device_count(), 1)
    print(f"World size {world_size}")
    config = Config()
    env = Wordle(word_restriction=config.word_restriction)
    config.word_to_index = env.dictionary_word_to_index
    config.index_to_word = env.dictionary_index_to_word

    training_params = {
        "resume": False,
        "epochs": 1000,
        "criterion": nn.NLLLoss,
        "load_path": "./weights/dynamics",
        "gpu1": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "gpu2": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    }
    mp.spawn(
        train_network,
        args=(
            data_dict,
            config,
            training_params,
        ),
        nprocs=world_size,
        join=True,
    )


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == "__main__":
    train_multi()
