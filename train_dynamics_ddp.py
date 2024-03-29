from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch

from ray.train.callbacks import JsonLoggerCallback
from ray.train import Trainer, TrainingCallback
from ray import train

import numpy as np
from prettytable import PrettyTable
from collections import deque
import os
import sys

from ML.networks import MuZeroNet
from ML.utils import load_weights
from config import Config
import numpy as np
from tqdm import tqdm

from globals import Embeddings, WordDictionaries
from wordle import Wordle
from config import Config


class Storage:
    def __init__(self, config, word_dictionary):
        self.word_dictionary = word_dictionary
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
            self.labels[self.n] = self.word_dictionary.result_index_dict[tuple(label)]
            self.states[self.n, :, :, :] = state
            self.n += 1
        return self.n >= self.max_n

    def save_state(self):
        np.save("word_data/actions", self.actions)
        np.save("word_data/labels", self.labels)
        np.save("word_data/states", self.states)

    def get_info(self):
        return self.states, self.actions, self.labels


def randomly_sample_games(shared_storage, config, word_dictionary):
    env = Wordle(word_dictionary)
    # TODO FIx progress bar
    for _ in tqdm(range(config.num_dynamics_examples)):
        states = []
        actions = []
        labels = []
        state, reward, done = env.reset()
        states.append(state.copy())
        while not done:
            action = np.random.randint(1, config.action_space + 1)
            print(word_dictionary.dictionary_word_to_index)
            print(action, word_dictionary.action_to_string(action))

            state, reward, done = env.step(word_dictionary.action_to_string(action))
            # Next batch
            labels.append(state[env.turn - 1, :, Embeddings.RESULT])
            actions.append(action)
            if not done:
                states.append(state.copy())

        full = shared_storage.save_game_history(states, actions, labels)
        if full:
            break


def create_dataset(num_examples=10000, num_workers=2):
    config = Config()
    word_dictionary = WordDictionaries(config.word_restriction)
    config.num_dynamics_examples = num_examples
    config.num_workers = num_workers
    shared_storage = Storage(config, word_dictionary)
    randomly_sample_games(shared_storage, config, word_dictionary)
    shared_storage.save_state()


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
        # major, minor = np.lib.format.read_magic(fhandle)
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
    def __init__(self, dataset_length):
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        label = torch.from_numpy(read_npy_chunk("word_data/labels.npy", index, 1))
        action = torch.from_numpy(
            read_npy_chunk("word_data/actions.npy", index, 1).astype(np.int16)
        )
        state = torch.from_numpy(read_npy_chunk("word_data/states.npy", index, 1))
        return (action, state, label)


class DynamicInMemory(Dataset):
    def __init__(self):
        self.labels = np.load("word_data/labels.npy")
        self.actions = np.load("word_data/actions.npy")
        self.states = np.load("word_data/states.npy")
        self.actions = self.actions.astype(np.int16)

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
        state = state.squeeze(1).long()
        label = label.squeeze(-1)
        print(action.shape, state.shape, label.shape)
        # print(action.get_device())
        # print(next(model.parameters()).device)
        pred = model(state, action)
        # print(pred.state_logprobs[0], label[0])
        loss = loss_fn(pred.state_logprobs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return {"loss": losses}


def train_func(inputs):
    config, word_dictionary = inputs
    training_data = DynamicInMemory()
    train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
    train_loader = train.torch.prepare_data_loader(train_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.train_on_gpu = False
    model = MuZeroNet(config, word_dictionary)._dynamics
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
        result = train_epoch(train_loader, model, loss_fn, optimizer)
        train.report(**result)
        results.append(result)

    return results


class PrintingCallback(TrainingCallback):
    def handle_result(self, results: list[dict], **info):
        print(results)


def train_cpu(num_workers=2, use_gpu=False):
    trainer = Trainer(
        backend="torch",
        num_workers=num_workers,
        use_gpu=use_gpu,
        logdir="./ray_\results",
    )
    config = Config()
    word_dictionary = WordDictionaries(config.word_restriction)
    trainer.start()
    results = trainer.run(train_func, (config, word_dictionary))
    trainer.shutdown()


def setup_world(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_network(id, data_dict, config, word_dictionary, training_params):
    print(f"Process {id}")
    model = MuZeroNet(config, word_dictionary)
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
    word_dictionary = WordDictionaries(config.word_restriction)

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
            word_dictionary,
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
    import argparse

    parser = argparse.ArgumentParser(
        description="""
        Create the dynamics function dataset and train the dynamics function on it.
        Requires gpus. 
        Requires a dataset to train.
        """
    )
    parser.add_argument(
        "-c",
        "--create-dataset",
        dest="create_dataset",
        help="Create the dataset",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    if args.create_dataset:
        create_dataset()
    else:
        if torch.cuda.is_available():
            train_multi()
        else:
            raise ValueError(f'Expected GPUs to be available, got None')
