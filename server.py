from multiprocessing.sharedctypes import Value
import os
import copy
import json
import logging
import numpy as np
from torch import set_grad_enabled
from torch import load
from torch import device as D
from collections import defaultdict
from MCTS_mu import MCTS
from ML.networks import MuZeroNet
from flask import Flask, jsonify, request
from flask_cors import CORS
from ML.utils import strip_module
from globals import PolicyOutputs
from wordle import Wordle
from config import Config
import torch

"""
API for wordle frontend
"""


class API(object):
    def __init__(self):
        self.seed = 1458
        self.config = Config()
        self.config.num_simulations = self.config.action_space * 5
        self.env = Wordle(word_restriction=self.config.action_space)
        self.config.word_to_index = self.env.dictionary_word_to_index
        self.config.index_to_word = self.env.dictionary_index_to_word
        self.model = MuZeroNet(self.config)
        self.load_model(self.model, self.config.production_path)
        self.reset()

    def load_model(self, model, path):
        if os.path.isfile(path):
            checkpoint = load(path, map_location=D("cpu"))
            model.load_state_dict(strip_module(checkpoint["weights"]))
            set_grad_enabled(False)
        else:
            raise ValueError("File does not exist")

    def model_inference(self, state: np.array, reward):
        root, mcts_info = MCTS(self.config).run(
            self.model,
            state,
            reward,
            self.env.turn,
        )
        visit_counts = np.array(
            [child.visit_count for child in root.children.values()],
            dtype="int32",
        )
        target_policy = visit_counts / np.sum(visit_counts)

        actions = [action for action in root.children.keys()]
        action = actions[np.argmax(visit_counts)]
        chosen_word = self.env.action_to_string(action)
        with torch.no_grad():
            outputs: PolicyOutputs = self.model.policy(
                torch.as_tensor(state).long().unsqueeze(0)
            )
        return outputs, target_policy, chosen_word, action

    def top_5_policy(self, policy):
        top5 = np.argpartition(policy[None, :], -5)[0][-5:]
        freqs = policy[top5]
        words = [self.env.action_to_string(num + 1) for num in top5]
        return {word: freq for word, freq in zip(words, freqs)}

    def step(self):
        model_outputs, target_policy, word, action = self.model_inference(
            self.current_state, self.reward
        )
        state, reward, done = self.env.step(word)
        if not done:
            model_outputs, target_policy, word, action = self.model_inference(
                state, self.reward
            )
        self.done = done
        self.reward = reward
        value = model_outputs.value.numpy().tolist()
        policy = self.top_5_policy(target_policy)
        return {
            "state": self.current_state.tolist(),
            "reward": reward,
            "done": done,
            "policy": json.dumps(policy),
            "value": value[0],
            "action": word,
            "turn": self.env.turn,
        }

    def reset(self):
        state, reward, done = self.env.reset()
        self.done = done
        self.reward = reward
        model_outputs, target_policy, word, action = self.model_inference(state, reward)
        value = model_outputs.value.numpy().tolist()
        policy = self.top_5_policy(target_policy)
        return {
            "state": state.tolist(),
            "reward": reward,
            "done": done,
            "policy": json.dumps(policy),
            "value": value[0],
            "action": word,
            "turn": self.env.turn,
        }

    @property
    def current_state(self):
        return self.env.state

    @property
    def target(self):
        return self.env.word


# instantiate env
api = API()

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"

cors = CORS(app, resources={r"/api/*": {"origins": "http://localhost:*"}})
cors = CORS(
    app, resources={r"/api/*": {"origins": "http://71.237.218.23*"}}
)  # This should be replaced with server public ip

logging.basicConfig(filename="logs/server.log", encoding="utf-8", level=logging.DEBUG)


@app.route("/health")
def home():
    return "Server is up and running"


@app.route("/api/player/name", methods=["POST"])
def player():
    req_data = json.loads(request.get_data())
    api.update_player_name(req_data.get("name"))
    return "Updated Name"


@app.route("/api/player/stats")
def player_stats():
    return json.dumps(api.return_player_stats())


@app.route("/api/model/outputs")
def model_outputs():
    return json.dumps(api.return_model_outputs())


@app.route("/api/model/load", methods=["POST"])
def load_model():
    req_data = json.loads(request.get_data())
    api.load_model(req_data.get("path"))
    return "Loaded Model"


@app.route("/api/reset")
def reset():
    return json.dumps(api.reset())


@app.route("/api/target", methods=["GET"])
def target():
    return json.dumps(api.target)


@app.route("/api/state")
def state():
    return json.dumps(api.state)


@app.route("/api/done")
def done():
    return json.dumps(api.done)


@app.route("/api/step", methods=["GET"])
def step():
    log = logging.getLogger(__name__)
    if api.done:
        api.reset()
    return json.dumps(api.step())


if __name__ == "__main__":
    app.run(debug=True, port=4000)
