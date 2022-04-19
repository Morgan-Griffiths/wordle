import ray
from ray.util.sgd.torch import TorchTrainer,TrainingOperator
import torch

ray.init()

trainer = TorchTrainer(
    training_operator_cls=CustomTrainingOperator,
    config={"lr": 0.01, # used in optimizer_creator
            "batch": 64 # used in data_creator
           },
    num_workers=2,  # amount of parallelism
    use_gpu=torch.cuda.is_available(),
    use_tqdm=True)