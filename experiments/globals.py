from torch.nn import CrossEntropyLoss, BCELoss, SmoothL1Loss
from ML.networks import Letters, Captials


class LearningCategories:
    MULTICLASS_CATEGORIZATION = "multiclass"
    BINARY_CATEGORIZATION = "binary"
    REGRESSION = "regression"


class DataTypes:
    CAPITALS = "capitals"
    LETTERS = "letters"


class NetworkConfig(object):
    DataModels = {
        DataTypes.CAPITALS: Captials,
        DataTypes.LETTERS: Letters,
    }
    LossFunctions = {
        LearningCategories.MULTICLASS_CATEGORIZATION: CrossEntropyLoss,
        LearningCategories.BINARY_CATEGORIZATION: BCELoss,
        LearningCategories.REGRESSION: SmoothL1Loss,
    }


dataMapping = {
    DataTypes.CAPITALS: LearningCategories.MULTICLASS_CATEGORIZATION,
    DataTypes.LETTERS: LearningCategories.MULTICLASS_CATEGORIZATION,
}

actionSpace = {DataTypes.LETTERS: 26, DataTypes.CAPITALS: 52}
