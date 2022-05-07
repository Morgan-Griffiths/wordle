from email.policy import default
from unittest import result
import numpy as np
from copy import deepcopy
from globals import (
    Embeddings,
    Tokens,
    Axis,
    alphabet,
    alphabet_dict,
    State,
    dictionary,
    index_to_letter_dict,
    readable_result_dict,
    target_dictionary,
)
from collections import defaultdict
from prettytable import PrettyTable


class Wordle:
    def __init__(self, word_restriction=None) -> None:
        super().__init__()
        if word_restriction is not None:
            step_size = len(dictionary) // word_restriction
            self.dictionary = dictionary[::step_size][:word_restriction]
            # self.dictionary = target_dictionary
        else:
            self.dictionary = dictionary
        # self.target_word_dictionary = target_dictionary
        self.alphabet_dict = alphabet_dict
        self.dictionary_word_to_index = {
            word: i for i, word in enumerate(self.dictionary,1)
        }
        self.dictionary_index_to_word = {
            i: word for i, word in enumerate(self.dictionary,1)
        }
        self.dictionary_index_to_word[0] = '-----'
        self.dictionary_word_to_index['-----'] = 0
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
            raise ValueError(f"{word.title()} is not contained in the dictionary")
        self.words.append(self.dictionary_word_to_index[word])
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
                state=update,
                letter=self.alphabet_dict[letter],
                turn=self.turn,
                word=word,
            )
            letter_result[i] = update
            letter_freqs[letter] = max(0, letter_freqs[letter] - 1)
        return letter_result

    def update_state(self, slot, state, letter, turn, word):
        # 5, 6, 3
        self.state[turn, slot, Embeddings.LETTER] = letter
        self.state[turn, slot, Embeddings.RESULT] = state
        self.state[turn, slot, Embeddings.WORD] = self.word_to_action(word)

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

    def action_to_string(self, action: int):
        return self.dictionary_index_to_word[action]

    def word_to_action(self, word: str):
        try:
            return self.dictionary_word_to_index[word.upper()]
        except:
            raise ValueError(f"Invalid word {word}")

    # def update_state(self, slot, state, letter, turn):
    #     # 5, 6, 26
    #     # turn emb, letter emb, state emb
    #     self.state[slot, turn, letter] = state
