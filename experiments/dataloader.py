from torch.utils.data import Dataset, DataLoader
import torch


class Data(Dataset):
    def __init__(self, X, y, category="classification"):
        self.X = X
        self.y = y
        self.category = category

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.category == "classification":
            sample = {
                "item": torch.tensor(self.X[idx]),
                "label": torch.tensor(self.y[idx]).long(),
            }
        else:
            sample = {
                "item": torch.tensor(self.X[idx]),
                "label": torch.tensor(self.y[idx]).float(),
            }

        return sample


def return_dataloader(X, y, category):
    data = Data(X, y, category)
    params = {"batch_size": 2048, "shuffle": True, "num_workers": 4}
    trainloader = DataLoader(data, **params)
    return trainloader
