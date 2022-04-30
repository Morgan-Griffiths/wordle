import copy
import os
import ray
import torch
from pathlib import Path


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)
        # print("IDS",ray.get_gpu_ids())
        # print('CUDA',torch.cuda.is_available())
    def save_checkpoint(self, path=None):
        if not path:
            path = os.path.join(self.config.weights_path, "model.checkpoint")
            Path(self.config.weights_path).mkdir(parents=True, exist_ok=True)

        torch.save(self.current_checkpoint, path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
