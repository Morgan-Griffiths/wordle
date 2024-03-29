import torch
from MCTS_mu import MCTS
from ML.networks import MuZeroNet
from ML.utils import strip_module, is_net_ddp
from globals import (
    DynamicOutputs,
    Embeddings,
    WordDictionaries,
    PolicyOutputs,
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

    def __init__(self, checkpoint, config, word_dictionary):
        self.config = config
        self.config.add_exploration_noise = False
        self.config.train_on_gpu = False
        self.word_dictionary = word_dictionary
        # Initialize the network
        self.model = MuZeroNet(self.config, self.word_dictionary)
        self.model.set_weights(strip_module(checkpoint["weights"]))
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
            print(f"policy_batch targets {policy_batch[0]}")
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
                        root, mcts_info = MCTS(self.config, self.word_dictionary).run(
                            self.model,
                            state,
                            reward,
                            env.turn,
                        )
                        visit_counts = np.array(
                            [child.visit_count for child in root.children.values()],
                            dtype="int32",
                        )
                        actions = [action for action in root.children.keys()]
                        action = actions[np.argmax(visit_counts)]
                        chosen_word = self.word_dictionary.action_to_string(action)
                        model_outputs: PolicyOutputs = self.model.policy(
                            torch.tensor(state.copy()).long().unsqueeze(0)
                        )
                        print(
                            f"model_outputs: {model_outputs.probs[:self.config.action_space]} {model_outputs.value}"
                        )
                        print("chosen_word", chosen_word)
                        print(
                            "highest prob word",
                            self.word_dictionary.action_to_string(
                                np.argmax(
                                    model_outputs.probs[
                                        : self.config.action_space
                                    ].numpy()
                                )
                                + 1
                            ),
                        )
                        dynamic_outputs: DynamicOutputs = self.model.dynamics(
                            torch.tensor(state.copy()).long().unsqueeze(0),
                            torch.tensor(action).view(1, 1),
                        )
                        state, reward, done = env.step(chosen_word)
                        print(env.visualize_state())
                        print(f"Next state {state[env.turn-1,:,Embeddings.RESULT]}")
                        result_index = self.word_dictionary.result_index_dict[
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

    def validate_mcts(self, env):
        try:
            with torch.no_grad():
                while True:
                    rewards = []
                    state, reward, done = env.reset()
                    while not done:
                        print(env.visualize_state())
                        root, mcts_info = MCTS(self.config, self.word_dictionary).run(
                            self.model, state, reward, env.turn
                        )
                        self.plot_mcts(root)
                        input(f"press ENTER to continue, or ctrl c to quit")

                        visit_counts = np.array(
                            [child.visit_count for child in root.children.values()],
                            dtype="int32",
                        )
                        actions = [action for action in root.children.keys()]
                        action = actions[np.argmax(visit_counts)]
                        chosen_word = self.word_dictionary.action_to_string(action)
                        state, reward, done = env.step(chosen_word)
                    print(env.visualize_state())
        except KeyboardInterrupt:
            return

    def plot_mcts(self, root, plot=True):
        """
        Plot the MCTS, pdf file is saved in the current directory.
        """
        try:
            from graphviz import Digraph
        except ModuleNotFoundError:
            print(
                "Please install graphviz to get the MCTS plot. e.g. brew install graphviz (OSX"
            )
            return None

        graph = Digraph(comment="MCTS", engine="neato")
        graph.attr("graph", rankdir="LR", splines="true", overlap="false")
        id = 0

        def traverse(node, action, parent_id, best, is_action: bool):
            nonlocal id
            node_id = id
            if is_action:
                graph.node(
                    str(node_id),
                    label=f"Action: {action}\nValue: {node.value:.2f}\nVisit count: {node.visit_count}\nPrior: {node.prior:.2f}\nReward: {node.reward:.2f}",
                    color="orange" if best else "black",
                )
            else:
                graph.node(
                    str(node_id),
                    label=f"State: {action}\nValue: {node.value:.2f}\nVisit count: {node.visit_count}\nPrior: {node.prior:.2f}\nReward: {node.reward:.2f}",
                    color="green" if best else "black",
                )
            id += 1
            if parent_id is not None:
                graph.edge(str(parent_id), str(node_id), constraint="false")

            if len(node.children) != 0:
                best_visit_count = max(
                    [child.visit_count for child in node.children.values()]
                )
            else:
                best_visit_count = False
            for action, child in node.children.items():
                if child.visit_count != 0:
                    traverse(
                        child,
                        action,
                        node_id,
                        True
                        if best_visit_count and child.visit_count == best_visit_count
                        else False,
                        not is_action,
                    )

        traverse(root, None, None, True, False)
        graph.node(str(0), color="red")
        graph.render("mcts", view=plot, cleanup=True, format="pdf")
        return graph
