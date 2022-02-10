from audioop import reverse
import sys
import os
import numpy as np
import torch
from config import Config
from ML.agents.base_agent import Agent
from ML.agents.mu_agent import MuAgent
from globals import (
    AgentData,
    Dims,
    NetworkOutput,
    Outputs,
    Results,
    State,
    dictionary,
    dictionary_word_to_index,
)
from utils import (
    DataStorage,
    DynamicsStorage,
    Stats,
    return_data_params,
    return_result_params,
    return_rewards,
    store_outputs,
    store_state,
    to_tensor,
    select_action,
)
from wordle import Wordle
from MCTS_mu import MCTS, Node
from prettytable import PrettyTable


def train_dynamics(env: Wordle, agent: MuAgent, training_params: dict) -> dict:
    config = Config()
    agent.train()
    loss = -1
    for e in range(training_params["epochs"]):
        storage = DynamicsStorage()
        sys.stdout.write("\r")
        state, reward, done = env.reset()
        while not done:
            action = torch.randint(0, Dims.OUTPUT, (1, 1))
            (
                state_prime,
                result,
                result_logits,
                reward,
                reward_logits,
            ) = agent.state_transition(to_tensor(state).unsqueeze(0), action.squeeze(0))
            storage.store_outputs(reward_logits, result_logits)
            state, reward, done = env.step(dictionary[action.item()])
            storage.store_result_target(state, env.turn)
        if env.turn < 2:
            rewards = [to_tensor(reward).unsqueeze(0)]
        else:
            rewards = [torch.tensor([0.0]) for _ in range(env.turn)]
            rewards[-1] = to_tensor(reward).unsqueeze(0)
        # rewards = return_rewards(env.turn, reward)
        storage.store_rewards(rewards)
        display_loss = agent.learn_dynamics(storage)
        sys.stdout.write(
            "[%-60s] %d%%"
            % (
                "=" * (60 * (e + 1) // training_params["epochs"]),
                (100 * (e + 1) // training_params["epochs"]),
            )
        )
        sys.stdout.flush()
        sys.stdout.write(f", epoch {e + 1}")
        sys.stdout.flush()
        sys.stdout.write(f", avg reward {display_loss}")
        sys.stdout.flush()
        if e % 100:
            evaluate_dynamics(env, agent)


def train_mcts(env: Wordle, agent: MuAgent, training_params: dict) -> dict:
    config = Config()
    training_results = return_result_params()
    agent.train()
    loss = -1
    for e in range(training_params["epochs"]):
        stats = Stats(config)
        data_params = DataStorage()
        sys.stdout.write("\r")
        state, reward, done = env.reset()
        # env.word = SET_WORD
        stats.store_state(state)
        data_params.store_state(state=state, done=done, word=env.word, turn=env.turn)
        while not done:
            root = Node(0)
            network_output = agent(state)
            root.expand(Dims.OUTPUT, network_output)
            root.add_exploration_noise(
                dirichlet_alpha=config.root_dirichlet_alpha,
                exploration_fraction=config.root_exploration_fraction,
            )
            action_history = MCTS(config).run(root, [], agent)
            action, visit_entropy = select_action(
                root, temperature=1, deterministic=False
            )
            (
                state_prime,
                result_logits,
                projected_reward,
                projected_rewards,
            ) = agent.state_transition(
                to_tensor(state).unsqueeze(0), to_tensor(action).unsqueeze(0)
            )
            outputs = NetworkOutput(
                value=network_output.value,
                result=network_output.result,
                reward=projected_reward,
                rewards=projected_rewards,
                policy_logits=network_output.policy_logits,
                state=state_prime,
                action=action,
                result_logits=result_logits,
            )
            data_params.store_outputs(outputs, action)
            stats.store_action(action)
            state, reward, done = env.step(dictionary[action])
            stats.store_state(state)
            stats.store_search_stats(root)
            data_params.store_state(
                state=state, done=done, word=env.word, turn=env.turn
            )
        with torch.no_grad():
            outputs: dict = agent(state)
        training_results[Results.ACTIONS].append(
            outputs.action.detach().squeeze(0).numpy()
        )
        training_results[Results.VALUES].append(
            outputs.value.detach().squeeze(0).numpy()
        )
        training_results[Results.ACTION_PROBS].append(
            outputs.policy_logits.detach().squeeze(0).numpy()
        )
        rewards = return_rewards(env.turn, reward)
        stats.store_rewards(rewards)
        target_values, target_rewards, target_policies = stats.make_target(
            0, env.turn, env.turn, agent.target_network
        )
        # data_params.store_outputs(outputs, outputs.action)
        data_params.store_rewards(rewards)
        agent.learn(data_params, target_values, target_rewards, target_policies)
        display_loss = (
            round(np.mean(training_results[Results.LOSSES]), 4)
            if training_results[Results.LOSSES]
            else 0
        )
        sys.stdout.write(
            "[%-60s] %d%%"
            % (
                "=" * (60 * (e + 1) // training_params["epochs"]),
                (100 * (e + 1) // training_params["epochs"]),
            )
        )
        sys.stdout.flush()
        sys.stdout.write(f", epoch {e + 1}")
        sys.stdout.flush()
        sys.stdout.write(f", avg reward {display_loss}")
        sys.stdout.flush()
        if e % training_params["eval_every"] == 0:
            agent.save_weights(
                os.path.join(training_params["save_dir"], training_params["agent_name"])
            )
            loss = evaluate(env, agent, training_params)
            training_results[Results.LOSSES].append(loss)
    training_results[Results.VALUES] = np.vstack(training_results[Results.VALUES])
    training_results[Results.ACTION_PROBS] = np.vstack(
        training_results[Results.ACTION_PROBS]
    )
    return training_results


def train(env: Wordle, agent: Agent, training_params: dict) -> dict:
    training_results = return_result_params()
    agent.train()
    loss = -1
    for e in range(training_params["epochs"]):
        data_params: dict = return_data_params()
        sys.stdout.write("\r")
        state, reward, done = env.reset()
        # env.word = SET_WORD
        data_params = store_state(data_params, state=state, done=done, target=env.word)
        while not done:
            outputs: dict = agent(state)
            data_params = store_outputs(data_params, outputs)
            state, reward, done = env.step(outputs[Outputs.WORD])
            data_params = store_state(
                data_params, state=state, done=done, target=env.word
            )
        rewards = return_rewards(env.turn, reward)
        store_training_results(training_results, outputs, env)
        data_params[AgentData.REWARDS].extend(rewards)
        agent.learn(data_params)
        display_loss = (
            round(np.mean(training_results[Results.LOSSES]), 4)
            if training_results[Results.LOSSES]
            else 0
        )
        sys.stdout.write(
            "[%-60s] %d%%"
            % (
                "=" * (60 * (e + 1) // training_params["epochs"]),
                (100 * (e + 1) // training_params["epochs"]),
            )
        )
        sys.stdout.flush()
        sys.stdout.write(f", epoch {e + 1}")
        sys.stdout.flush()
        sys.stdout.write(f", avg reward {display_loss}")
        sys.stdout.flush()
        if e % training_params["eval_every"] == 0:
            # check_agent(data_params, agent, env)
            agent.save_weights(
                os.path.join(training_params["save_dir"], training_params["agent_name"])
            )
            loss = evaluate(env, agent, training_params)
            training_results[Results.LOSSES].append(loss)
    training_results[Results.VALUES] = np.vstack(training_results[Results.VALUES])
    training_results[Results.ACTION_PROBS] = np.vstack(
        training_results[Results.ACTION_PROBS]
    )
    return training_results


def store_training_results(training_results, outputs, env):
    training_results[Results.ACTIONS].append(
        outputs[Outputs.ACTION].detach().squeeze(0).numpy()
    )
    training_results[Results.VALUES].append(
        outputs[Outputs.VALUES].detach().squeeze(0).numpy()
    )
    training_results[Results.ACTION_PROBS].append(
        outputs[Outputs.ACTION_PROBS].detach().squeeze(0).numpy()
    )
    training_results[Results.TARGETS].append(dictionary_word_to_index[env.word])
    return training_results


def evaluate(env, agent, training_params):
    agent.evaluate()
    loss = []
    for e in range(training_params["evaluation_epochs"]):
        state, reward, done = env.reset()
        # env.word = SET_WORD
        while not done:
            outputs: DataStorage = agent(state)
            word = dictionary[outputs.action]
            state, reward, done = env.step(word)
        loss.append(reward)
    return np.mean(loss)


def check_agent(data_params: dict, agent: Agent, env: Wordle):
    env.visualize_state()
    visualize_probabilities(data_params)
    visualize_new_probabilities(agent, env)
    visualize_q_values(data_params)
    visualize_new_q_values(agent, env)
    # visualize_values(data_params)
    # visualize_new_values(agent, env)


def evaluate_dynamics(env, agent: MuAgent):
    # goal here is to visualize how the distribution changes over time given a scenario
    agent.evaluate()
    rewards = []
    results = []
    table = PrettyTable(["projected_result", "result", "projected_reward", "reward"])
    state, reward, done = env.reset()
    env.word = dictionary[4]
    actions = [0]
    actions.extend(list(range(5)))
    for a in actions:
        action = torch.tensor(a)
        word = dictionary[action.item()]
        (
            state_prime,
            result,
            result_logits,
            projected_reward,
            reward_logits,
        ) = agent.state_transition(to_tensor(state).unsqueeze(0), action.unsqueeze(0))
        rewards.append(projected_reward)
        results.append(result_logits)
        state, reward, done = env.step(word)
        target = state[env.turn - 1, :, 1].astype(int)
        table.add_row([result, target, projected_reward.detach().numpy(), reward])
    print(table)


def visualize_probabilities(data_params: dict):
    action_probs = data_params[Results.ACTION_PROBS]
    table = PrettyTable(dictionary)
    for probs in action_probs:
        table.add_row(probs.detach().squeeze(0).numpy())
    print("\nold probabilities")
    print(table)


def visualize_new_probabilities(agent: Agent, env: Wordle):
    table = PrettyTable(dictionary)
    current_state = np.zeros(State.SHAPE)
    for turn in range(env.turn):
        current_state[turn, :] = env.state[turn, :]
        outputs = agent(current_state)
        table.add_row(outputs[Outputs.ACTION_PROBS].detach().squeeze(0).numpy())
    print("\nnew probabilities")
    print(table)


def visualize_values(data_params: dict):
    values = data_params[Results.VALUES]
    actions = data_params[Results.ACTIONS]
    table = PrettyTable([f"V(s)", "Action"])
    for act, val in zip(actions, values):
        table.add_row([val.squeeze(0).item(), act.squeeze(0).item()])
    print("\nold values")
    print(table)


def visualize_new_values(agent: Agent, env: Wordle):
    table = PrettyTable([f"V(s)", "Action"])
    current_state = np.zeros(State.SHAPE)
    for turn in range(env.turn):
        current_state[turn, :] = env.state[turn, :]
        outputs = agent(current_state)
        table.add_row(
            [
                outputs[Outputs.VALUES].squeeze(0).item(),
                outputs[Outputs.ACTION].squeeze(0).item(),
            ]
        )
    print("\nnew values")
    print(table)


def visualize_q_values(data_params: dict):
    values = data_params[Results.VALUES]
    table = PrettyTable(dictionary)
    for val in values:
        table.add_row(val.detach().squeeze(0).numpy())
    print("\nold values")
    print(table)


def visualize_new_q_values(agent: Agent, env: Wordle):
    table = PrettyTable(dictionary)
    current_state = np.zeros(State.SHAPE)
    for turn in range(env.turn):
        current_state[turn, :] = env.state[turn, :]
        outputs = agent(current_state)
        table.add_row(outputs[Outputs.VALUES].detach().squeeze(0).numpy())
    print("\nnew values")
    print(table)
