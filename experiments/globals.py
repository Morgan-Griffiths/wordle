from torch.nn import CrossEntropyLoss, BCELoss, SmoothL1Loss
from ML.networks import Letters, Captials, Reinforce, WordleTest, StateActionTransition


class LearningCategories:
    MULTICLASS_CATEGORIZATION = "multiclass"
    BINARY_CATEGORIZATION = "binary"
    REGRESSION = "regression"


class DataTypes:
    CAPITALS = "capitals"
    LETTERS = "letters"
    WORDLE = "wordle"
    REINFORCE = "reinforce"
    MULTI_TARGET = "multitarget"
    CONSTELLATION = "constellation"


class NetworkConfig(object):
    DataModels = {
        DataTypes.CAPITALS: Captials,
        DataTypes.LETTERS: Letters,
        DataTypes.WORDLE: StateActionTransition,
        DataTypes.REINFORCE: Reinforce,
        DataTypes.MULTI_TARGET: WordleTest,
        DataTypes.CONSTELLATION: WordleTest,
    }
    LossFunctions = {
        LearningCategories.MULTICLASS_CATEGORIZATION: CrossEntropyLoss,
        LearningCategories.BINARY_CATEGORIZATION: BCELoss,
        LearningCategories.REGRESSION: SmoothL1Loss,
    }


dataMapping = {
    DataTypes.CAPITALS: LearningCategories.MULTICLASS_CATEGORIZATION,
    DataTypes.LETTERS: LearningCategories.MULTICLASS_CATEGORIZATION,
    DataTypes.WORDLE: LearningCategories.MULTICLASS_CATEGORIZATION,
    DataTypes.REINFORCE: LearningCategories.MULTICLASS_CATEGORIZATION,
    DataTypes.MULTI_TARGET: LearningCategories.MULTICLASS_CATEGORIZATION,
    DataTypes.CONSTELLATION: LearningCategories.MULTICLASS_CATEGORIZATION,
}

actionSpace = {
    DataTypes.LETTERS: 26,
    DataTypes.CAPITALS: 52,
    DataTypes.WORDLE: 5,
    DataTypes.REINFORCE: 5,
    DataTypes.MULTI_TARGET: 5,
    DataTypes.CONSTELLATION: 5,
}
