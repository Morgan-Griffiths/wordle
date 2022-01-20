import torch.nn as nn
import torch
import numpy as np
from globals import State
from utils import to_tensor

fake_input = np.zeros(State.SHAPE)
fake_input = to_tensor(fake_input).unsqueeze(0)  # .type(torch.LongTensor)
# fake_input = torch.LongTensor(fake_input)
print(fake_input.shape)
print(fake_input.dtype)
# fake_input = torsch.zeros(State.SHAPE)
embed = nn.Embedding(4, 16)
conv1 = nn.Conv3d(6, 16, kernel_size=(5, 26, 16), stride=(1, 1, 1), bias=False)

x = embed(fake_input)
res = conv1(x)
print(res.shape)

index = torch.tensor([0, 1, 2]).long()
# index = torch.tensor([[0, 0], [1, 1], [2, 2]]).long()
b = torch.arange(9).view(3, 3)

print(b[index, index], b[index, index].shape)
