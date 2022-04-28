import ray
from ray.util.sgd.torch import TorchTrainer, TrainingOperator
import torch

ray.init()

CustomTrainingOperator = TrainingOperator.from_creators(
    model_creator=ResNet18,  # A function that returns a nn.Module
    optimizer_creator=optimizer_creator,  # A function that returns an optimizer
    data_creator=cifar_creator,  # A function that returns dataloaders
    loss_creator=torch.nn.CrossEntropyLoss,  # A loss function
)

trainer = TorchTrainer(
    training_operator_cls=CustomTrainingOperator,
    config={
        "lr": 0.01,  # used in optimizer_creator
        "batch": 64,  # used in data_creator
    },
    num_workers=2,  # amount of parallelism
    use_gpu=torch.cuda.is_available(),
    use_tqdm=True,
)

stats = trainer.train()
print(trainer.validate())

torch.save(trainer.state_dict(), "checkpoint.pt")
trainer.shutdown()
print("success!")
