from dataclasses import dataclass
from typing import Any
import numpy as np
import os
from copy import copy
from pyrsistent import v
import torch
from ML.agents.base_agent import Agent
from ML.agents.mu_agent import MuAgent
from globals import Dims, Outputs, dictionary
from wordle import Wordle

""" 
Action selection
a_t = argmax (Q(s,a) + U(s,a)) 
C: exploration rate
U: C(s) * P(s,a) * sqrt(N(s)) / 1 + N(s,a)
N(s) = parent visit count
C(s) = log((1 + N(s) + C_base)/C_base) + c_init
W(s) = total action value
"""
EPS = 1e-8


class MCTS:
    def __init__(self, game, agent) -> None:
        self.game: Wordle = game
        self.agent: Agent = agent
        self.reset()

    def reset(self):
        self.num_sims = 20
        self.c = 1
        self.c_init = 1
        self.actions = {i: set() for i in range(6)}

        self.Wsa = {}
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s

    def getActionProb(self, game, temp=0.8):
        for _ in range(self.num_sims):
            self.search(game.copy())

        s = str(game.words)

        counts = np.zeros(Dims.OUTPUT)
        for a in self.actions:
            if (s, a) in self.Nsa:
                counts[a] = self.Nsa[(s, a)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, game):
        ps = game.state
        s = str(game.words)

        if s not in self.Es:
            self.Es[s] = game.rewards
        if game.game_over:
            return self.Es[s]
        with torch.no_grad():
            outputs = self.agent(ps)
        if s not in self.Ps:
            self.Ps[s] = outputs[Outputs.ACTION_PROBS].squeeze(0)
            self.Ns[s] = 0
        cur_best = -float("inf")
        best_act = -1
        self.actions[game.turn].add(outputs[Outputs.ACTION].item())
        for a in self.actions[game.turn]:
            # self.c = np.log((1 + self.Ns[s] + self.c) / self.c) + self.c_init
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.c * self.Ps[s][a] * (
                    np.sqrt(self.Ns[s]) / 1 + self.Nsa[(s, a)]
                )
            else:
                u = float("inf")
                # u = self.c + self.Ps[s][a] * np.sqrt(self.Ns[s])
            if u > cur_best:
                cur_best = u
                best_act = a

        word = dictionary[best_act]
        game.step(word)
        v = self.search(game)

        if (s, best_act) in self.Qsa:
            self.Qsa[(s, best_act)] = (
                self.Nsa[(s, best_act)] * self.Qsa[(s, best_act)] + v
            ) / (self.Nsa[(s, best_act)] + 1)
            self.Nsa[(s, best_act)] += 1
        else:
            self.Nsa[(s, best_act)] = 1
            self.Qsa[(s, best_act)] = v

        self.Ns[s] += 1
        return v


# argmax_a [Q(s,a) + P(s,a) * sqrt(N(s)/1+N(s,a)) * (c1 * log(N(s) + c2 + 1) / c2)]
# temp decays. first half 1, 2nd quarter 0.5 last quarter 0.25


class MU_MCTS:
    def __init__(self, game, agent) -> None:
        self.game: Wordle = game
        self.agent: MuAgent = agent
        self.reset()

    def reset(self):
        self.num_sims = 50
        self.c1 = 1.25
        self.c2 = 19652
        self.actions = {i: set() for i in range(6)}

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Ssa = {}  # internal state
        self.Rsa = {}  # rewards

        self.Es = {}  # stores game.getGameEnded ended for board s

    def expansion(self):
        state = self.game.state
        outputs = self.agent.query_policy(state)
        s_prime, r_prime = self.agent.state_transition(
            state, outputs[Outputs.ACTION], self.game.turn
        )
        a, v = self.policy(s_prime)

    def getActionProb(self, game, temp=1.0):
        for _ in range(self.num_sims):
            self.search(game.copy())

        s = str(game.words)

        counts = np.zeros(Dims.OUTPUT)
        for a in self.actions:
            if (s, a) in self.Nsa:
                counts[a] = self.Nsa[(s, a)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, game):
        ps = game.state
        s = str(game.words)

        if s not in self.Es:
            self.Es[s] = game.rewards
        if game.game_over:
            return self.Es[s]
        with torch.no_grad():
            outputs = self.agent(ps)
        if s not in self.Ps:
            self.Ps[s] = outputs[Outputs.ACTION_PROBS].squeeze(0)
            self.Ns[s] = 0
        cur_best = -float("inf")
        best_act = -1
        self.actions[game.turn].add(outputs[Outputs.ACTION].item())
        for a in self.actions[game.turn]:
            # self.c = np.log((1 + self.Ns[s] + self.c) / self.c) + self.c_init
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.c * self.Ps[s][a] * (
                    np.sqrt(self.Ns[s]) / 1 + self.Nsa[(s, a)]
                )
            else:
                u = float("inf")
                # u = self.c + self.Ps[s][a] * np.sqrt(self.Ns[s])
            if u > cur_best:
                cur_best = u
                best_act = a

        word = dictionary[best_act]
        game.step(word)
        v = self.search(game)

        if (s, best_act) in self.Qsa:
            self.Qsa[(s, best_act)] = (
                self.Nsa[(s, best_act)] * self.Qsa[(s, best_act)] + v
            ) / (self.Nsa[(s, best_act)] + 1)
            self.Nsa[(s, best_act)] += 1
        else:
            self.Nsa[(s, best_act)] = 1
            self.Qsa[(s, best_act)] = v

        self.Ns[s] += 1
        return v
