import numpy as np
from copy import deepcopy
from globals import Embeddings, Tokens, alphabet, State, Mappings
from collections import defaultdict
from prettytable import PrettyTable


class Wordle:
    def __init__(self, mappings: Mappings) -> None:
        super().__init__()
        self.mappings = mappings
        self.gamma = 0.05
        self.reset()

    def reset(self):
        self.word = np.random.choice(self.mappings.target_dictionary)
        self.alphabet = {letter: Tokens.UNKNOWN for letter in alphabet}
        self.turn = 0
        self.state = np.zeros(State.SHAPE, dtype=np.int8)
        self.game_over = False
        self.rewards = 0
        self.words = []
        return self.state, self.rewards, self.game_over

    def step(self, word):
        word = word.lower()
        if word not in self.mappings.dictionary_word_to_index:
            raise ValueError(f"{word.title()} is not contained in the dictionary")
        self.words.append(self.mappings.dictionary_word_to_index[word])
        result = self.evaluate_word(word)
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
        """0:unknown, 1: not contained, 2: wrong spot, 3: right spot"""
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
            self.update_state(
                slot=i,
                result=update,
                letter=self.mappings.alphabet_dict[letter],
                turn=self.turn,
            )
            letter_result[i] = update
            letter_freqs[letter] = max(0, letter_freqs[letter] - 1)
        return letter_result

    def update_state(self, slot, result, letter, turn):
        # 5, 6, 2
        self.state[turn, slot, Embeddings.LETTER] = letter
        self.state[turn, slot, Embeddings.RESULT] = result

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
                row_items.append(self.mappings.index_to_letter_dict[letter])
                row_items.append(self.mappings.readable_result_dict[int(result)])
            table.add_row(row_items)
        print(f"Target word {self.word}")
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
