import copy
import re
import time

import numpy as np
import ray
import torch

from ML.networks import MuZeroNet
from globals import Embeddings, PolicyOutputs, result_index_dict, State


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        np.random.seed(self.config.seed)

    def save_game(self, game_history, shared_storage=None):
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = np.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                        np.abs(root_value - self.compute_target_value(game_history, i))
                        ** self.config.PER_alpha
                    )
                    priorities.append(priority)

                game_history.priorities = np.array(priorities, dtype="float32")
                game_history.game_priority = np.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:

            shared_storage.set_info.remote(
                {
                    "episode_length": len(game_history.action_history) - 1,
                    "actions": np.array(game_history.max_actions),
                    "total_reward": sum(game_history.reward_history),
                    "mean_value": np.mean(
                        [value for value in game_history.root_values if value]
                    ),
                    "num_played_games": self.num_played_games,
                    "num_played_steps": self.num_played_steps,
                }
            )

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        (
            index_batch,
            state_batch,
            action_batch,
            reward_batch,
            result_batch,
            value_batch,
            policy_batch,
            word_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        for game_id, game_history, game_prob in self.sample_n_games(
            self.config.batch_size
        ):
            game_pos, pos_prob = self.sample_position(game_history)
            (
                states,
                values,
                rewards,
                policies,
                actions,
                result_targets,
                word_targets,
            ) = self.make_target(game_history, game_pos)

            index_batch.append([game_id, game_pos])
            state_batch.append(states)
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            result_batch.append(result_targets)
            word_batch.append(word_targets)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )
            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = np.array(weight_batch, dtype="float32") / max(weight_batch)
        state_batch = np.array(state_batch)
        state_batch = (
            torch.tensor(state_batch).long().view(self.config.batch_size, *State.SHAPE)
        )
        value_batch = torch.tensor(value_batch).float()
        reward_batch = torch.tensor(reward_batch).float()
        policy_batch = (
            torch.tensor(policy_batch)
            .float()
            .view(self.config.batch_size, self.config.action_space)
        )
        action_batch = torch.tensor(action_batch).long().view(self.config.batch_size)
        result_batch = torch.tensor(result_batch).long().view(self.config.batch_size)
        word_batch = torch.tensor(word_batch).long().view(self.config.batch_size)
        assert state_batch.shape == (self.config.batch_size, *State.SHAPE)
        assert value_batch.shape == (self.config.batch_size, 1)
        assert reward_batch.shape == (self.config.batch_size, 1)
        assert policy_batch.shape == (self.config.batch_size, self.config.action_space)
        assert result_batch.dim() == 1
        assert action_batch.dim() == 1
        assert word_batch.dim() == 1
        # state_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
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
            ),
        )

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = np.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= np.sum(game_probs)
            game_index = np.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = np.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)
            game_probs = np.array(game_probs, dtype="float32")
            game_probs /= np.sum(game_probs)
            game_prob_dict = dict(
                [(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)]
            )
            selected_games = np.random.choice(game_id_list, n_games, p=game_probs)
        else:
            selected_games = np.random.choice(list(self.buffer.keys()), n_games)
            game_prob_dict = {}
        ret = [
            (game_id, self.buffer[game_id], game_prob_dict.get(game_id))
            for game_id in selected_games
        ]
        return ret

    def sample_position(self, game_history, force_uniform=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)
            position_index = np.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = np.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = np.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update game priorities
                self.buffer[game_id].game_priority = np.max(
                    self.buffer[game_id].priorities
                )

    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        # bootstrap_index = index + self.config.td_steps
        # if bootstrap_index < len(game_history.root_values):
        #     root_values = (
        #         game_history.root_values
        #         if game_history.reanalysed_predicted_root_values is None
        #         else game_history.reanalysed_predicted_root_values
        #     )
        #     last_step_value = root_values[bootstrap_index]

        #     value = last_step_value * self.config.discount_rate ** self.config.td_steps
        # else:
        #     value = 0
        end_reward = game_history.reward_history[-1]
        value = end_reward * self.config.discount_rate ** (
            len(game_history.reward_history) - index
        )
        return value

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        (
            target_states,
            target_values,
            target_rewards,
            target_policies,
            actions,
            result_targets,
            word_targets,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        # for current_index in range(0, len(game_history.action_history)):
        #     value = self.compute_target_value(game_history, current_index)

        # if current_index < len(game_history.root_values):
        value = self.compute_target_value(game_history, state_index)
        target_values.append(value)
        assert (
            game_history.state_history[state_index].shape == State.SHAPE
        ), f"Expected {State.SHAPE} got {game_history.state_history[state_index].shape}"
        target_states.append(game_history.state_history[state_index])
        target_rewards.append(game_history.reward_history[state_index])
        target_policies.append(game_history.child_visits[state_index])
        actions.append(game_history.action_history[state_index])
        word_targets.append(game_history.word_history[state_index])
        result_targets.append(
            result_index_dict[tuple(game_history.result_history[state_index])]
        )
        # elif current_index == len(game_history.root_values):
        #     target_values.append(0)
        #     target_rewards.append(game_history.reward_history[current_index])
        #     # Uniform policy
        #     target_policies.append(
        #         [
        #             1 / len(game_history.child_visits[0])
        #             for _ in range(len(game_history.child_visits[0]))
        #         ]
        #     )
        #     actions.append(game_history.action_history[current_index])
        # else:
        #     # States past the end of games are treated as absorbing states
        #     target_values.append(0)
        #     target_rewards.append(0)
        #     # Uniform policy
        #     target_policies.append(
        #         [
        #             1 / len(game_history.child_visits[0])
        #             for _ in range(len(game_history.child_visits[0]))
        #         ]
        #     )
        #     actions.append(np.random.choice(self.config.action_space))
        return (
            target_states,
            target_values,
            target_rewards,
            target_policies,
            actions,
            result_targets,
            word_targets,
        )


@ray.remote
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = MuZeroNet(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        # self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu")) # uncomment for gpu
        self.model.eval()

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

    def reanalyse(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            game_id, game_history, _ = ray.get(
                replay_buffer.sample_game.remote(force_uniform=True)
            )

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            if self.config.use_last_model_value:
                observations = np.array(
                    [
                        game_history.state_history[i]
                        for i in range(len(game_history.root_values))
                    ]
                )
                observations = (
                    torch.tensor(observations)
                    .long()
                    .to(next(self.model.parameters()).device)
                )
                policy_outputs: PolicyOutputs = self.model.policy(observations)
                game_history.reanalysed_predicted_root_values = (
                    torch.squeeze(policy_outputs.value).detach().cpu().numpy()
                )

            replay_buffer.update_game_history.remote(game_id, game_history)
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )
