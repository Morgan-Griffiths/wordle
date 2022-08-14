import numpy as np
from tqdm import tqdm

from globals import Embeddings, Mappings
from wordle import Wordle
from config import Config


class Storage:
    def __init__(self, config, mappings):
        self.mapping = mappings
        self.actions = np.zeros(shape=(config.num_dynamics_examples), dtype=np.uint16)
        self.states = np.zeros(
            shape=(config.num_dynamics_examples, 6, 5, 2), dtype=np.uint8
        )
        self.labels = np.zeros(shape=(config.num_dynamics_examples), dtype=np.uint8)
        self.n = 0
        self.max_n = config.num_dynamics_examples

    def save_game_history(self, states, actions, labels):
        for state, action, label in zip(states, actions, labels):
            if self.n >= self.max_n:
                break
            self.actions[self.n] = action
            self.labels[self.n] = self.mapping.result_index_dict[tuple(label)]
            self.states[self.n, :, :, :] = state
            self.n += 1
        return self.n >= self.max_n

    def save_state(self):
        np.save("word_data/actions", self.actions)
        np.save("word_data/labels", self.labels)
        np.save("word_data/states", self.states)

    def get_info(self):
        return self.states, self.actions, self.labels


def randomly_sample_games(shared_storage, config):
    mappings = Mappings(config.word_restriction)
    env = Wordle(mappings)
    for _ in tqdm(range(config.num_dynamics_examples)):
        states = []
        actions = []
        labels = []
        state, reward, done = env.reset()
        states.append(state.copy())
        while not done:
            action = np.random.randint(1, config.action_space + 1)
            state, reward, done = env.step(mappings.action_to_string(action))
            # Next batch
            labels.append(state[env.turn - 1, :, Embeddings.RESULT])
            actions.append(action)
            if not done:
                states.append(state.copy())

        full = shared_storage.save_game_history(states, actions, labels)
        if full:
            break


def create_dataset(num_examples=10000, num_workers=2):
    config = Config()
    mappings = Mappings(config.word_restriction)
    config.num_dynamics_examples = num_examples
    config.num_workers = num_workers
    shared_storage = Storage(config, mappings)
    randomly_sample_games(shared_storage, config)
    shared_storage.save_state()
