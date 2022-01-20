from email.policy import default
import numpy as np
from globals import Tokens, alphabet, Axis, alphabet_dict, State, dictionary
from collections import defaultdict


class Wordle:
    def __init__(self) -> None:
        super().__init__()
        self.dictionary = dictionary
        self.alphabet_dict = alphabet_dict
        self.gamma = 0.05
        self.reset()

    def reset(self):
        self.word = np.random.choice(self.dictionary)
        self.alphabet = {letter: Tokens.UNKNOWN for letter in alphabet}
        self.turn = 0
        self.state = np.zeros(State.SHAPE)
        return self.state, 0, False

    def step(self, word):
        word = word.upper()
        if word not in self.dictionary:
            raise ValueError("Not a real word. Nice try bozo")
        result = self.evaluate_word(word)
        self.update_alphabet(word, result)
        self.increment_turn()
        game_over = self.done(result)
        rewards = self.reward(result, game_over)
        return self.state, rewards, game_over

    def increment_turn(self):
        self.turn += 1

    def update_alphabet(self, word, result):
        for i, letter in enumerate(word):
            self.alphabet[letter] = result[i]

    def reward(self, result, game_over):
        if sum(result) == Tokens.EXACT * 5 and game_over:
            return 1 - (self.gamma * (self.turn - 1))
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
            self.update_state(
                slot=i,
                state=update,
                letter=self.alphabet_dict[letter],
                turn=self.turn,
            )
            letter_result[i] = update
            letter_freqs[letter] = max(0, letter_freqs[letter] - 1)
        return letter_result

    def update_state(self, slot, state, letter, turn):
        # 6, 5, 26
        self.state[turn, slot, letter] = state
