import numpy as np
import itertools as it
import os
import json
from scipy.stats import entropy
from globals import dictionary

MISSING = np.uint8(0)
CONTAINED = np.uint8(1)
EXACT = np.uint8(2)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)
PATTERN_GRID_DATA = dict()
PATTERN_MATRIX_FILE = os.path.join(DATA_DIR, "pattern_matrix.npy")
PATTERN_MATRIX_FILE_JULIA = os.path.join(DATA_DIR, "pattern_matrix.npz")
SHORT_WORD_LIST_FILE = os.path.join(DATA_DIR, "possible_words.txt")
LONG_WORD_LIST_FILE = os.path.join(DATA_DIR, "allowed_words.txt")
WORD_FREQ_FILE = os.path.join(DATA_DIR, "wordle_words_freqs_full.txt")
WORD_FREQ_MAP_FILE = os.path.join(DATA_DIR, "freq_map.json")


def sigmoid(x):
    return 1 / 1 + np.e ** (-x)


def words_to_int_arrays(words):
    return np.array([[ord(c) for c in w] for w in words], dtype=np.uint8)


def get_word_frequencies(regenerate=False):
    if os.path.exists(WORD_FREQ_MAP_FILE) or regenerate:
        with open(WORD_FREQ_MAP_FILE) as fp:
            result = json.load(fp)
        return result
    # Otherwise, regenerate
    freq_map = dict()
    with open(WORD_FREQ_FILE) as fp:
        for line in fp.readlines():
            pieces = line.split(" ")
            word = pieces[0]
            freqs = [float(piece.strip()) for piece in pieces[1:]]
            freq_map[word] = np.mean(freqs[-5:])
    with open(WORD_FREQ_MAP_FILE, "w") as fp:
        json.dump(freq_map, fp)
    return freq_map


def get_word_list(short=False):
    result = []
    file = SHORT_WORD_LIST_FILE if short else LONG_WORD_LIST_FILE
    with open(file) as fp:
        result.extend([word.strip() for word in fp.readlines()])
    return result


def generate_pattern_matrix(words1, words2):
    """
    A pattern for two words represents the worle-similarity
    pattern (grey -> 1, yellow -> 2, green -> 3) but as an integer
    between 0 and 3^5. Reading this integer in ternary gives the
    associated pattern.
    This function computes the pairwise patterns between two lists
    of words, returning the result as a grid of hash values. Since
    this can be time-consuming, many operations that can be are vectorized
    (perhaps at the expense of easier readibility), and the the result
    is saved to file so that this only needs to be evaluated once, and
    all remaining pattern matching is a lookup
    """

    # Number of letters/words
    nl = len(words1[0])
    nw1 = len(words1)  # Number of words
    nw2 = len(words2)  # Number of words

    # Convert word lists to integer arrays
    word_arr1, word_arr2 = map(words_to_int_arrays, (words1, words2))

    # equality_grid keeps track of all equalities between all pairs
    # of letters in words. Specifically, equality_grid[a, b, i, j]
    # is true when words[i][a] == words[b][j]
    equality_grid = np.zeros((nw1, nw2, nl, nl), dtype=bool)
    for i, j in it.product(range(nl), range(nl)):
        equality_grid[:, :, i, j] = np.equal.outer(word_arr1[:, i], word_arr2[:, j])

    # full_pattern_matrix[a, b] should represent the 5-color pattern
    # for guess a and answer b, with 0 -> grey, 1 -> yellow, 2 -> green
    full_pattern_matrix = np.zeros((nw1, nw2, nl), dtype=np.uint8)

    # Green pass
    for i in range(nl):
        matches = equality_grid[
            :, :, i, i
        ].flatten()  # matches[a, b] is true when words[a][i] = words[b][i]
        full_pattern_matrix[:, :, i].flat[matches] = EXACT

        for k in range(nl):
            # If it's a match, mark all elements associated with
            # that letter, both from the guess and answer, as covered.
            # That way, it won't trigger the yellow pass.
            equality_grid[:, :, k, i].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False

    # Yellow pass
    for i, j in it.product(range(nl), range(nl)):
        matches = equality_grid[:, :, i, j].flatten()
        full_pattern_matrix[:, :, i].flat[matches] = CONTAINED
        for k in range(nl):
            # Similar to above, we want to mark this letter
            # as taken care of, both for answer and guess
            equality_grid[:, :, k, j].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False

    # Rather than representing a color pattern as a lists of integers,
    # store it as a single integer, whose ternary representations corresponds
    # to that list of integers.
    pattern_matrix = np.dot(full_pattern_matrix, (3 ** np.arange(nl)).astype(np.uint8))

    return pattern_matrix


def generate_full_pattern_matrix():
    words = get_word_list()
    pattern_matrix = generate_pattern_matrix(words, words)
    # Save to file
    np.save(PATTERN_MATRIX_FILE, pattern_matrix)
    # np.savez(PATTERN_MATRIX_FILE_JULIA, pattern_matrix)
    return pattern_matrix


def get_pattern_matrix(words1, words2):
    if not PATTERN_GRID_DATA:
        if not os.path.exists(PATTERN_MATRIX_FILE):
            print(
                "\n".join(
                    [
                        "Generating pattern matrix. This takes a minute, but",
                        "the result will be saved to file so that it only",
                        "needs to be computed once.",
                    ]
                )
            )
            generate_full_pattern_matrix()
        PATTERN_GRID_DATA["grid"] = np.load(PATTERN_MATRIX_FILE)
        PATTERN_GRID_DATA["words_to_index"] = dict(zip(get_word_list(), it.count()))

    full_grid = PATTERN_GRID_DATA["grid"]
    words_to_index = PATTERN_GRID_DATA["words_to_index"]

    indices1 = [words_to_index[w] for w in words1]
    indices2 = [words_to_index[w] for w in words2]
    mesh = np.ix_(indices1, indices2)
    return full_grid[mesh]


def get_pattern_distributions(allowed_words: list, possible_words: list, weights: list):
    """
    For each possible guess in allowed_words, this finds the probability
    distribution across all of the 3^5 wordle patterns you could see, assuming
    the possible answers are in possible_words with associated probabilities
    in weights.
    It considers the pattern hash grid between the two lists of words, and uses
    that to bucket together words from possible_words which would produce
    the same pattern, adding together their corresponding probabilities.
    """
    pattern_matrix = get_pattern_matrix(allowed_words, possible_words)
    n = len(allowed_words)
    distributions = np.zeros((n, 3 ** 5))
    n_range = np.arange(n)
    for j, prob in enumerate(weights):
        distributions[n_range, pattern_matrix[:, j]] += prob
    return distributions


def get_frequency_based_priors(n_common=3000, width_under_sigmoid=10):
    """
    We know that that list of wordle answers was curated by some human
    based on whether they're sufficiently common. This function aims
    to associate each word with the likelihood that it would actually
    be selected for the final answer.
    Sort the words by frequency, then apply a sigmoid along it.
    """
    freq_map = get_word_frequencies()
    words = np.array(list(freq_map.keys()))
    freqs = np.array([freq_map[w] for w in words])
    arg_sort = freqs.argsort()
    sorted_words = words[arg_sort]

    # We want to imagine taking this sorted list, and putting it on a number
    # line so that it's length is 10, situating it so that the n_common most common
    # words are positive, then applying a sigmoid
    x_width = width_under_sigmoid
    c = x_width * (-0.5 + n_common / len(words))
    xs = np.linspace(c - x_width / 2, c + x_width / 2, len(words))
    priors = dict()
    for word, x in zip(sorted_words, xs):
        priors[word] = sigmoid(x)
    return priors


def get_true_wordle_prior():
    words = get_word_list()
    short_words = get_word_list(short=True)
    return dict((w, int(w in short_words)) for w in words)


def get_possible_words(guess, pattern, word_list):
    all_patterns = get_pattern_matrix([guess], word_list).flatten()
    return list(np.array(word_list)[all_patterns == pattern])


def get_weights(words, priors):
    frequencies = np.array([priors[word] for word in words])
    total = frequencies.sum()
    if total == 0:
        return np.zeros(frequencies.shape)
    return frequencies / total


def entropy_of_distributions(distributions, atol=1e-12):
    axis = len(distributions.shape) - 1
    return entropy(distributions, base=2, axis=axis)


def get_entropies(allowed_words, possible_words, weights):
    if weights.sum() == 0:
        return np.zeros(len(allowed_words))
    distributions = get_pattern_distributions(allowed_words, possible_words, weights)
    return entropy_of_distributions(distributions)


def entropy_to_expected_score(ent):
    """
    Based on a regression associating entropies with typical scores
    from that point forward in simulated games, this function returns
    what the expected number of guesses required will be in a game where
    there's a given amount of entropy in the remaining possibilities.
    """
    # Assuming you can definitely get it in the next guess,
    # this is the expected score
    min_score = 2 ** (-ent) + 2 * (1 - 2 ** (-ent))

    # To account for the likely uncertainty after the next guess,
    # and knowing that entropy of 11.5 bits seems to have average
    # score of 3.5, we add a line to account
    # we add a line which connects (0, 0) to (3.5, 11.5)
    return min_score + 1.5 * ent / 11.5


# arr = np.load(PATTERN_MATRIX_FILE)
# print(arr, arr.dtype)
# print(type(arr), type(arr[0]))
# generate_full_pattern_matrix()
allowed_words = get_word_list(short=False)
possible_words = allowed_words
priors = get_frequency_based_priors()
weights = get_weights(allowed_words, priors)
H0 = entropy_of_distributions(weights)
H1s = get_entropies(allowed_words, possible_words, weights)
word_to_weight = dict(zip(possible_words, weights))
probs = np.array([word_to_weight.get(w, 0) for w in allowed_words])
expected_scores = probs + (1 - probs) * (1 + entropy_to_expected_score(H0 - H1s))
best_guess = np.argmin(expected_scores)
print(f"best guess {allowed_words[best_guess]}")
# print(len(weights))
# a = [
#     "MOUNT",
#     "HELLO",
#     "NIXED",
#     "AAHED",
#     "HELMS",
# ]
# result = get_pattern_matrix(a, a)
# print(result)
# patterns = get_pattern_distributions(words, words, weights)


# If this guess is the true answer, score is 1. Otherwise, it's 1 plus
# the expected number of guesses it will take after getting the corresponding
# amount of information.
# print(patterns)
