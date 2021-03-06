import numpy as np
from copy import deepcopy
from globals import (
    Embeddings,
    Tokens,
    alphabet,
    Axis,
    alphabet_dict,
    State,
    dictionary,
    dictionary_word_to_index,
    index_to_letter_dict,
    readable_result_dict,
)
from collections import defaultdict
from prettytable import PrettyTable


class Wordle:
    def __init__(self) -> None:
        super().__init__()
        self.dictionary = dictionary
        self.alphabet_dict = alphabet_dict
        self.dictionary_word_to_index = dictionary_word_to_index
        self.gamma = 0.05
        self.reset()

    def reset(self):
        self.word = np.random.choice(self.dictionary)
        self.alphabet = {letter: Tokens.UNKNOWN for letter in alphabet}
        self.turn = 0
        self.state = np.zeros(State.SHAPE)
        self.game_over = False
        self.rewards = 0
        self.words = []
        return self.state, self.rewards, self.game_over

    def step(self, word):
        word = word.upper()
        if word not in self.dictionary_word_to_index:
            raise ValueError("Not a real word. Nice try bozo")
        self.words.append(self.dictionary_word_to_index[word])
        result = self.evaluate_word(word)
        self.update_state(result)
        self.update_alphabet(word, result)
        self.increment_turn()
        self.game_over = self.done(result)
        self.rewards = self.reward(result, self.game_over)
        return self.state, self.rewards, self.game_over

    def increment_turn(self):
        self.turn += 1

    def update_alphabet(self, word, result):
        for i, letter in enumerate(word):
            self.alphabet[letter] = result[i]

    def reward(self, result, game_over):
        if sum(result) == Tokens.EXACT * 5 and game_over:
            return 1  # - (self.gamma * (self.turn - 1))
        elif game_over:
            return -1
        return 0

    def done(self, result):
        if sum(result) == Tokens.EXACT * 5 or self.turn >= 6:
            return True
        return False

    def evaluate_word(self, word):
        """0: not contained, 1: wrong spot, 2: right spot, 3:unknown"""
        letter_result = np.zeros(5)
        letter_freqs = defaultdict(lambda: 0)
        for letter in self.word:
            letter_freqs[letter] += 1
        for i, letter in enumerate(word):
            if letter_freqs[letter] > 0:
                if letter == self.word[i]:
                    update = Tokens.EXACT
                elif letter in self.word:
                    update = Tokens.CONTAINED
                else:
                    update = Tokens.MISSING
            else:
                update = Tokens.MISSING
            letter_result[i] = update
            letter_freqs[letter] = max(0, letter_freqs[letter] - 1)
        return letter_result

    def update_state(self, result):
        # 6, 5, 2
        self.state[self.turn, :, :] = result

    def visualize_state(self):
        letter_headers = [f"Letters_{i}" for i in range(5)]
        result_headers = [f"Results_{i}" for i in range(5)]
        headers = letter_headers.extend(result_headers)
        table = PrettyTable(headers)
        for turn in range(6):
            row_items = []
            for row in range(5):
                letter = self.state[turn, row, Embeddings.LETTER]
                result = self.state[turn, row, Embeddings.RESULT]
                row_items.append(index_to_letter_dict[letter])
                row_items.append(readable_result_dict[int(result)])
            table.add_row(row_items)
        print(self.word)
        print(table)

    def copy(self):
        env = Wordle()
        env.word = deepcopy(self.word)
        env.state = deepcopy(self.state)
        env.game_over = deepcopy(self.game_over)
        env.rewards = deepcopy(self.rewards)
        env.turn = deepcopy(self.turn)
        env.words = deepcopy(self.words)
        return env

    # def update_state(self, slot, state, letter, turn):
    #     # 5, 6, 26
    #     # turn emb, letter emb, state emb
    #     self.state[slot, turn, letter] = state
