import torch
from MCTS_mu import MCTS
from ML.networks import MuZeroNet
from globals import (
    DynamicOutputs,
    Embeddings,
    PolicyOutputs,
    dictionary_index_to_word,
    result_index_dict,
)
import numpy as np
import ray


class ValidateModel:
    """
    Tools to understand the learned model.
    Args:
        weights: weights for the model to validate.
        config: configuration class instance related to the weights.
    """

    def __init__(self, checkpoint, config):
        self.config = config

        # Initialize the network
        self.model = MuZeroNet(self.config)
        self.model.set_weights(checkpoint["weights"])
        self.model.eval()

    def check_model_updates(self, trainer, per_buffer, shared_storage):
        try:
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
            print(f"dynamic targets {result_batch}")
            while True:

                actor_prior: PolicyOutputs = self.model.policy(state_batch)
                dynamic_prior: DynamicOutputs = self.model.dynamics(
                    state_batch, action_batch.unsqueeze(1)
                )
                trainer.test.remote(batch, shared_storage)
                self.model.set_weights(
                    ray.get(shared_storage.get_info.remote("weights"))
                )
                actor_post: PolicyOutputs = self.model.policy(state_batch)
                dynamic_post: DynamicOutputs = self.model.dynamics(
                    state_batch, action_batch.unsqueeze(1)
                )
                print(
                    f"actor_prior. Probs: {actor_prior.probs[0]}, Value {actor_prior.value[0]}"
                )
                print(
                    f"actor_post. Probs: {actor_post.probs[0]}, Value {actor_post.value[0]}"
                )
                print(f"dynamic_prior. Probs: {dynamic_prior.state_probs[0][-40:]}")
                print(f"dynamic_post. Probs: {dynamic_post.state_probs[0][-40:]}")
                input(f"press ENTER to continue, or ctrl c to quit")
        except KeyboardInterrupt:
            return

    def validate(self, env) -> None:
        print("validating model")
        try:
            with torch.no_grad():
                rewards = []
                while True:
                    state, reward, done = env.reset()
                    print(env.visualize_state())
                    model_outputs: PolicyOutputs = self.model.policy(
                        torch.tensor(state.copy()).long().unsqueeze(0)
                    )
                    while not done:
                        root, mcts_info = MCTS(self.config).run(
                            self.model,
                            state,
                            reward,
                        )
                        visit_counts = np.array(
                            [child.visit_count for child in root.children.values()],
                            dtype="int32",
                        )
                        actions = [action for action in root.children.keys()]
                        action = actions[np.argmax(visit_counts)]
                        chosen_word = env.action_to_string(action)
                        model_outputs: PolicyOutputs = self.model.policy(
                            torch.tensor(state.copy()).long().unsqueeze(0)
                        )
                        print(
                            f"model_outputs: {model_outputs.probs[:self.config.action_space]} {model_outputs.value}"
                        )
                        print("chosen_word", chosen_word)
                        dynamic_outputs: DynamicOutputs = self.model.dynamics(
                            torch.tensor(state.copy()).long().unsqueeze(0),
                            torch.tensor(action).view(1, 1),
                        )
                        state, reward, done = env.step(chosen_word)
                        print(env.visualize_state())
                        print(f"Next state {state[env.turn-1,:,Embeddings.RESULT]}")
                        result_index = result_index_dict[
                            tuple(state[env.turn - 1, :, Embeddings.RESULT])
                        ]
                        print(
                            f"Prob of actual state {dynamic_outputs.state_probs.squeeze(0)[result_index]}, prob of winning state {dynamic_outputs.state_probs.squeeze(0)[-1]}"
                        )
                        input(f"press ENTER to continue, or ctrl c to quit")
                        if done:
                            rewards.append(reward)
                    print(f"total_reward {rewards}")

        except KeyboardInterrupt:
            return
