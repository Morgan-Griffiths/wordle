from collections import defaultdict
from information_theory.information import filter_words, information
from information_theory.wordle import Wordle
from globals import dictionary, result_index_dict, index_result_dict
import numpy as np
from plot import plot_hist
import sys
import time


def create_first_order_result_distributions():
    tic = time.time()
    env = Wordle()
    word_result_dist = {}
    word_entropy_dist = defaultdict(lambda: [])
    starting_information = information(len(dictionary), 1)
    print("starting_information", starting_information)
    for i, word in enumerate(dictionary):
        sys.stdout.write(f"\r")
        env.word = word
        # hist_results = []
        result_matrix = np.zeros(3 ** 5)
        for guess in dictionary:
            result = env.evaluate_word(guess)
            # hist_results.append(result_index_dict[tuple(result)])
            remaining_words = filter_words(dictionary, guess, result)
            remaining_probability = information(len(dictionary), len(remaining_words))
            result_matrix[result_index_dict[tuple(result)]] += 1
            word_entropy_dist[guess].append(
                starting_information - remaining_probability
            )
        word_result_dist[word] = result_matrix
        sys.stdout.write(
            "[%-60s] %d%%"
            % (
                "=" * (60 * (i + 1) // len(dictionary)),
                (100 * (i + 1) // len(dictionary)),
            )
        )
        sys.stdout.flush()
        sys.stdout.write(f", epoch {i + 1}")
        sys.stdout.flush()
        sys.stdout.write(
            f", time remaining {(len(dictionary) / (i + 1)) * ((time.time() - tic)/60)}"
        )
        sys.stdout.flush()
    best_word = ""
    most_entropy = 0
    for word in dictionary:
        word_entropy_dist[word] = np.mean(word_entropy_dist[word])
        if word_entropy_dist[word] > most_entropy:
            best_word = word
            most_entropy = word_entropy_dist[word]
    return word_result_dist, word_entropy_dist, best_word, most_entropy


def main():
    env = Wordle()
    env.word = dictionary[0]
    starting_information = information(len(dictionary), 1)
    print("starting_information", starting_information)
    guessed_word = dictionary[50]
    result = env.evaluate_word(guessed_word)
    remaining_words = filter_words(dictionary, guessed_word, result)
    remaining_probability = information(len(dictionary), len(remaining_words))
    print("remaining_probability", remaining_probability)
    print("information gained", starting_information - remaining_probability)


(
    word_result_dist,
    word_entropy_dist,
    best_word,
    most_entropy,
) = create_first_order_result_distributions()
# plot_hist(
#     "AAHED",
#     word_dist["AAHED"],
# )
print(best_word)
print(most_entropy)
