from torch.nn import CrossEntropyLoss, BCELoss, SmoothL1Loss
from ML.networks import Threshold, StateActionTransition, ZeroPolicy


class LearningCategories:
    MULTICLASS_CATEGORIZATION = "multiclass"
    BINARY_CATEGORIZATION = "binary"
    REGRESSION = "regression"


class DataTypes:
    THRESHOLD = "threshold"
    WORDLE = "wordle"
    RANDOM = "random"
    POLICY = "policy"


class NetworkConfig(object):
    DataModels = {
        DataTypes.THRESHOLD: Threshold,
        DataTypes.WORDLE: StateActionTransition,
        DataTypes.RANDOM: StateActionTransition,
        DataTypes.POLICY: ZeroPolicy,
    }
    LossFunctions = {
        LearningCategories.MULTICLASS_CATEGORIZATION: CrossEntropyLoss,
        LearningCategories.BINARY_CATEGORIZATION: BCELoss,
        LearningCategories.REGRESSION: SmoothL1Loss,
    }


dataMapping = {
    DataTypes.THRESHOLD: LearningCategories.MULTICLASS_CATEGORIZATION,
    DataTypes.WORDLE: LearningCategories.MULTICLASS_CATEGORIZATION,
    DataTypes.RANDOM: LearningCategories.MULTICLASS_CATEGORIZATION,
    DataTypes.POLICY: LearningCategories.MULTICLASS_CATEGORIZATION,
}

actionSpace = {
    DataTypes.THRESHOLD: 7,
    DataTypes.WORDLE: 5,
    DataTypes.RANDOM: 5,
    DataTypes.POLICY: 5,
}
