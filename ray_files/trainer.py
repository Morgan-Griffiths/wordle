import time
import ray
import torch
import numpy as np
import copy
from ML.networks import MuZeroNet, dict_to_cpu
import torch.nn.functional as F

from globals import DynamicOutputs, PolicyOutputs, State


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.config.train_on_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = 'cpu'
        # Initialize the network
        self.model = MuZeroNet(config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(self.device)
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
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_rate
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            (
                priorities,
                actor_loss,
                dynamic_loss,
                policy_loss,
                value_loss,
            ) = self.update_weights(batch, shared_storage)

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "actor_loss": actor_loss,
                    "dynamic_loss": dynamic_loss,
                    "total_loss": actor_loss + dynamic_loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)

    def update_weights(self, batch, shared_storage):
        """
        Perform one training step.
        """
        # WEIGHT Update
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
        assert state_batch.shape == (self.config.batch_size, *State.SHAPE)
        assert value_batch.shape == (self.config.batch_size, 1)
        assert reward_batch.shape == (self.config.batch_size, 1)
        assert policy_batch.shape == (self.config.batch_size, self.config.action_space)
        assert result_batch.dim() == 1
        assert action_batch.dim() == 1

        device = next(self.model.parameters()).device
        state_batch = state_batch.to(device) 
        action_batch = action_batch.to(device) 
        value_batch = value_batch.to(device) 
        reward_batch = reward_batch.to(device) 
        policy_batch = policy_batch.to(device)
        result_batch = result_batch.to(device) 
        word_batch = word_batch.to(device)

        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        dynamics_outputs: DynamicOutputs = self.model.dynamics(
            state_batch,
            action_batch.unsqueeze(1),
        )
        dynamic_loss = F.nll_loss(dynamics_outputs.state_logprobs, result_batch)
        # policy update
        policy_outputs: PolicyOutputs = self.model.policy(state_batch)
        # policy_loss = F.nll_loss(policy_outputs.logprobs, word_batch, reduction="none")
        policy_loss = (-policy_batch * policy_outputs.logprobs).sum(1)
        value_loss = F.smooth_l1_loss(
            reward_batch, policy_outputs.value, reduction="none"
        )
        actor_loss = policy_loss + value_loss

        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            actor_loss *= weight_batch
        loss = actor_loss.sum() + dynamic_loss
        self.optimizer.zero_grad()
        # loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model._policy.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self.training_step += 1
        # Priority update
        target_value_scalar = np.array(reward_batch.cpu(), dtype="float32")
        priorities = np.zeros_like(target_value_scalar)
        priorities = (
            np.abs(policy_outputs.value.detach().cpu().numpy() - target_value_scalar)
            ** self.config.PER_alpha
        )
        shared_storage.set_info.remote(
            {
                "actor_probs": policy_outputs.probs[0],
                "actor_value": policy_outputs.value[0],
                "dynamic_prob_winning_state": dynamics_outputs.state_probs[0],
                "results": result_batch[0],
            }
        )
        return (
            priorities,
            actor_loss.mean().item(),
            dynamic_loss.mean().item(),
            policy_loss.mean().item(),
            value_loss.mean().item(),
        )

    def test(self, batch, shared_storage):
        (
            priorities,
            actor_loss,
            dynamic_loss,
            policy_loss,
            value_loss,
        ) = self.update_weights(batch, shared_storage)
        shared_storage.set_info.remote(
            {
                "weights": copy.deepcopy(self.model.cpu().get_weights()),
                "optimizer_state": copy.deepcopy(
                    dict_to_cpu(self.optimizer.state_dict())
                ),
            }
        )
