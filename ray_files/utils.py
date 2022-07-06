from ML.networks import MuZeroNet
import ray
import torch


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = MuZeroNet(config)
        if config.load_dynamic_weights:
            print(f"Loading dynamic weights from  {config.dynamics_weight_path}")
            checkpoint = torch.load(config.dynamics_weight_path, map_location="cpu")
            dynamic_dict = model._dynamics.state_dict()
            pretrained_dict = {k[10:]:v for k, v in checkpoint['weights'].items() if k.find('_dynamics') > -1}
            model._dynamics.load_state_dict(pretrained_dict)
        weights = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weights, summary