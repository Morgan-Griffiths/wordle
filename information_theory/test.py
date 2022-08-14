from information_theory.wordle import Wordle
from information_theory.information import filter_words
import numpy as np


def test_filter(env: Wordle, word_dictionary):
    env.word = word_dictionary.dictionary[50]
    guessed_word = word_dictionary.dictionary[66]
    result = env.evaluate_word(guessed_word)
    remaining_words = filter_words(word_dictionary.dictionary, guessed_word, result)
    assert remaining_words == [
        "ABHOR",
        "ABORD",
        "ABORT",
        "ABRAM",
        "ABRAY",
        "ABRIM",
        "ABRIN",
        "ABRIS",
        "AMBRY",
        "ARABA",
        "ARBAS",
        "ARBOR",
        "AROBA",
    ]
    assert np.array_equal(result, np.array([3.0, 1.0, 1.0, 2.0, 2.0]))
