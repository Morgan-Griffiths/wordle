from re import T
from ML.agents.base_agent import Agent
from ML.agents.q_agent import Q_agent
from ML.agents.ppo import PPO_agent
from ML.agents.p_agent import P_agent
from ML.agents.mu_agent import MuAgent
from config import Config
from wordle import Wordle
from globals import Models, Outputs, Results, Tokens, Dims, AgentData, Train, dictionary
import torch
from plot import plot_data, plot_frequencies, plot_q_values
import sys
import os
import numpy as np
from utils import shape_values_to_q_values, store_outputs, store_state, return_rewards
from train import train, train_dynamics, train_mcts

SET_WORD = "AALII"


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


def return_agent(model):
    config = Config()
    params = {"nA": Dims.OUTPUT}
    if model == Models.Q_LEARNING:
        agent = Q_agent(params, config)
    elif model == Models.AC_LEARNING:
        agent = P_agent(params, config)
    elif model == Models.PPO:
        agent = PPO_agent(params, config)
    elif model == Models.POLICY:
        agent = P_agent(params, config)
    elif model == Models.MUZERO:
        agent = MuAgent(config)
    return agent


def main(resume, model, epochs: int, train_type: str):
    env = Wordle()
    agent = return_agent(model)
    if resume:
        agent.load_weights("weights/test")
    training_params = {
        "epochs": epochs,
        "evaluation_epochs": 250,
        "eval_every": 50,
        "save_dir": "weights",
        "agent_name": "test",
    }
    if train_type == Train.MCTS:
        training_results: dict = train_mcts(env, agent, training_params)
    elif train_type == Train.REGULAR:
        training_results: dict = train(env, agent, training_params)
    elif train_type == Train.DYNAMICS:
        training_results: dict = train_dynamics(env, agent, training_params)
    else:
        raise ValueError(f"Unknown training type {train_type}")

    plot_data("Training reward", [training_results[Results.LOSSES]], "loss")
    plot_frequencies(
        "Word freqs over time", training_results[Results.ACTION_PROBS], dictionary
    )
    data = shape_values_to_q_values(
        training_results[Results.VALUES], training_results[Results.ACTIONS]
    )
    # plot_q_values("Values over time", data, dictionary)


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
        metavar=f"[{Models.Q_LEARNING},{Models.AC_LEARNING},{Models.PPO},{Models.POLICY},{Models.MUZERO}]",
        default=Models.MUZERO,
        help="which model",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        dest="epochs",
        default=5000,
        type=int,
        help="training epochs",
    )
    parser.add_argument(
        "--train_type",
        "-t",
        dest="train_type",
        default=Train.DYNAMICS,
        type=str,
        metavar=f"{Train.REGULAR},{Train.MCTS},{Train.DYNAMICS}",
        help="training style",
    )
    args = parser.parse_args()

    main(args.resume, args.model, args.epochs, args.train_type)
