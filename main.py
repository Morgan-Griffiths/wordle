from re import T
from ML.agent import Agent
from wordle import Wordle
from ML.networks import Policy
from globals import Models, Outputs, Results, Tokens, Dims, AgentData, dictionary
import torch
from plot import plot_data, plot_frequencies, plot_values
import sys
import os
import numpy as np
from utils import store_outputs, store_state, return_rewards


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


def main(resume, model):
    seed = 1234
    env = Wordle()
    agent = Agent(Dims.OUTPUT, seed, model)
    if resume:
        agent.load_weights("weights/test")
    training_params = {
        "epochs": 5000,
        "evaluation_epochs": 100,
        "eval_every": 50,
        "save_dir": "weights",
        "agent_name": "test",
    }
    training_results: dict = train(env, agent, training_params)

    plot_data("Training reward", [training_results[Results.LOSSES]], "loss")
    plot_frequencies(
        "Word freqs over time", training_results[Results.ACTION_PROBS], dictionary
    )
    plot_values("Values over time", training_results[Results.VALUES], dictionary)


def train(env: Wordle, agent: Agent, training_params: dict) -> dict:
    training_results = {
        Results.LOSSES: [],
        Results.VALUES: [],
        Results.ACTION_PROBS: [],
    }
    agent.train()
    loss = -1
    for e in range(training_params["epochs"]):
        data_params = {
            AgentData.STATES: [],
            AgentData.ACTIONS: [],
            AgentData.ACTION_PROBS: [],
            AgentData.VALUES: [],
            AgentData.REWARDS: [],
            AgentData.DONES: [],
        }
        sys.stdout.write("\r")
        state, reward, done = env.reset()
        env.word = "AAHED"
        data_params = store_state(data_params, state=state, done=done)
        while not done:
            outputs: dict = agent(state)
            data_params = store_outputs(data_params, outputs)
            state, reward, done = env.step(outputs[Outputs.WORD])
            data_params = store_state(data_params, state=state, done=done)
        training_results[Results.VALUES].append(
            outputs[Outputs.VALUES].detach().squeeze(0).numpy()
        )
        training_results[Results.ACTION_PROBS].append(
            outputs[Outputs.ACTION_PROBS].detach().squeeze(0).numpy()
        )
        rewards = return_rewards(env.turn, reward)
        data_params[AgentData.REWARDS].extend(rewards)
        agent.learn(data_params)
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
        sys.stdout.write(f", avg reward {round(np.mean(training_results['losses']),4)}")
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


def evaluate(env, agent, training_params):
    agent.evaluate()
    loss = []
    for e in range(training_params["evaluation_epochs"]):
        state, reward, done = env.reset()
        while not done:
            outputs: dict = agent(state)
            state, reward, done = env.step(outputs["word"])
        loss.append(reward)
    return np.mean(loss)


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
        "--model",
        "-m",
        dest="model",
        metavar="[q_learning,ac_learning]",
        default=Models.Q_LEARNING,
        help="which model",
    )
    args = parser.parse_args()

    main(args.resume, args.model)
