from collections import defaultdict
import numpy as np

from information_theory.wordle import Wordle
from globals import dictionary, Tokens

""" 
E[Information] = sum([prob(x) for x in words] * information(x))
information = - log2(p)
 """


def filter_words(current_words: list, guessed_word: str, results: list) -> list:
    remaining_words = []
    for word in current_words:
        for i, (letter, result) in enumerate(zip(guessed_word, results)):
            if result == Tokens.MISSING and letter in word:
                break
            elif result == Tokens.EXACT and letter != word[i]:
                break
            elif result == Tokens.CONTAINED and letter not in word:
                break
            elif i == 4:
                remaining_words.append(word)
    return remaining_words


def information(current_words: int, remaining_words: int):
    if remaining_words == 0:
        return 0
    probability = remaining_words / current_words
    return -np.log2(probability)


def result_distribution(word_choice):
    ...


def remaining_information():
    ...
