import numpy as np
from ML.networks import Letters
from experiments.globals import DataTypes

alphabet = "".join("abcdefghijklmnopqrstuvwxyz".split())
alphabet_dict = {i: letter for i, letter in enumerate(alphabet)}


def generate_letter_mapping_data():
    inputs = []
    targets = []
    for i in range(26):
        vector = np.zeros(26)
        vector[i] = 1
        targets.append(i)
        inputs.append(vector)
    return np.vstack(inputs), np.vstack(targets)


def generate_capital_letter_mapping_data():
    inputs = []
    targets = []
    for i in range(26):  # lowercase
        vector = np.zeros(26)
        vector[i] = 1
        targets.append(i)
        inputs.append(vector)
    for i in range(26):  # capitals
        vector = np.zeros(26)
        vector[i] = 2
        targets.append(i + 26)
        inputs.append(vector)
    return np.vstack(inputs), np.vstack(targets)


def load_data(datatype):
    if datatype == DataTypes.CAPITALS:
        x, y = generate_capital_letter_mapping_data()
    elif datatype == DataTypes.LETTERS:
        x, y = generate_letter_mapping_data()
    return {
        "trainX": x,
        "trainY": y,
        "valX": x,
        "valY": y,
    }
