class Tokens:
    UNKNOWN = 0
    MISSING = 1
    CONTAINED = 2
    EXACT = 3


class Results:
    VALUES = "values"
    ACTION_PROBS = "action_probs"
    LOSSES = "losses"


class Axis:
    SLOT = 0
    STATE = 1
    LETTER = 2
    TURN = 3


class Models:
    Q_LEARNING = "q_learning"
    AC_LEARNING = "ac_learning"


class Outputs:
    VALUES = "values"
    ACTION_PROBS = "action_probs"
    ACTION_PROB = "action_prob"
    ACTION = "action"
    WORD = "word"


class AgentData:
    REWARDS = "rewards"
    ACTIONS = "actions"
    ACTION_PROBS = "action_probs"
    VALUES = "values"
    STATES = "states"
    DONES = "dones"


class State:
    SHAPE = (6, 5, 26)


class Dims:
    INPUT = 26
    OUTPUT = 5  # 12972


with open("wordle.txt", "r") as f:
    wordle_dictionary = f.readlines()
dictionary = [word.strip() for word in wordle_dictionary[: Dims.OUTPUT]]
print(dictionary)
alphabet = "".join("abcdefghijklmnopqrstuvwxzy".upper().split())
alphabet_dict = {letter: i for i, letter in enumerate(alphabet)}
