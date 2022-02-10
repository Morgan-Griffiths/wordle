from information_theory.wordle import Wordle
from information_theory.information import filter_words
from globals import dictionary

# dictionary = ['HELLO','HELMS','HEALS','SLATE','CRANE']

test_words = ["OOOOO", "HATER"]


def test_filter():
    env = Wordle()
    env.word = dictionary[50]
    guessed_word = dictionary[66]
    result = env.evaluate_word(guessed_word)
    remaining_words = filter_words(dictionary, guessed_word, result)
    print(dictionary)
    print(guessed_word)
    print(result)
    print(remaining_words)


# def test_filter():
#     filter_words(dictionary,)

test_filter()
