import ray
import torch
import numpy
import copy
from ML.networks import ZeroPolicy, StateActionTransition
import torch.nn.functional as F

from globals import DynamicOutputs


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = StateActionTransition()
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.train()
        self.training_step = initial_checkpoint["training_step"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr_init,
            weight_decay=self.config.weight_decay,
        )
        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

def update_lr(self):
    lr =  self.config.lr_init * self.lr_decay_rate ** (self.training_step / self.config.lr_decay_rate)
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr


def update_weights(self, batch):
    """
    Perform one training step.
    """
    (states, actions, results) = batch
    dynamics_outputs: DynamicOutputs = self.model.dynamics(
        states,
        actions,
    )
    dynamic_loss = (
        F.cross_entropy(dynamics_outputs.state_probs, results, reduction="sum")
        * self.config.value_loss_weight
    )
    self.optimizer.zero_grad()
    dynamic_loss.backward()
    self.optimizer.step()
    self.training_step += 1
    return dynamic_loss.item()
