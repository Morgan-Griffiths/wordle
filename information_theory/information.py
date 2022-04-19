from collections import defaultdict
import numpy as np

from information_theory.wordle import Wordle
from globals import Tokens

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

def filter_words_numpy(word_arr: np.array, guessed_word: str, results: list) -> list:
    remaining_words = []
    for i, (letter, result) in enumerate(zip(guessed_word, results)):
        if result == Tokens.MISSING:
            # remove all words
            a = word_arr[:, 0] != letter
            b = word_arr[:, 1] != letter
            c = word_arr[:, 2] != letter
            d = word_arr[:, 3] != letter
            e = word_arr[:, 4] != letter
            mask = a & b & c & d & e
            word_arr = word_arr[mask]
        elif result == Tokens.EXACT:
            word_arr = word_arr[word_arr[:, i] == letter]
        elif result == Tokens.CONTAINED:
            numbers = [num for num in range(5) if num != i]
            a = word_arr[:, numbers[0]] == letter
            b = word_arr[:, numbers[1]] == letter
            c = word_arr[:, numbers[2]] == letter
            d = word_arr[:, numbers[3]] == letter
            mask = a | b | c | d
            word_arr = word_arr[mask]
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
