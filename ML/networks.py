import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *
import numpy as np
from globals import Outputs, Tokens
from torch.distributions import Categorical


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).float()  # .to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class Q_learning(nn.Module):
    def __init__(self, seed, output_dims):
        super(Q_learning, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.emb = nn.Embedding(Tokens.EXACT + 1, 16, padding_idx=0)
        self.conv1 = nn.Conv3d(
            6, 16, kernel_size=(5, 26, 16), stride=(1, 1, 1), bias=False
        )
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 128)

        # the part for the value function
        self.value_output = nn.Linear(128, 1)
        self.advantage_output = nn.Linear(128, output_dims)
        self.noise = GaussianNoise()

    def forward(self, x: torch.LongTensor):
        # B,26
        B = x.shape[0]
        y = self.emb(x)
        y = F.leaky_relu(self.conv1(y))
        # 6,16,1,1
        y = F.leaky_relu(self.fc1(y.squeeze().squeeze()))
        y = F.leaky_relu(self.fc2(y.view(B, -1)))
        a = self.advantage_output(y)
        v = self.value_output(y)
        v = v.expand_as(a)
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        # pick action
        action_logits = self.noise(q)
        action_probs = F.softmax(action_logits, dim=-1)
        m = Categorical(action_probs)
        action = m.sample()
        action_prob = m.log_prob(action)
        return {
            Outputs.ACTION: action,
            Outputs.ACTION_PROB: action_prob,
            Outputs.ACTION_PROBS: action_probs,
            Outputs.VALUES: F.tanh(q),
        }


class Policy(nn.Module):
    def __init__(self, seed, output_dims):
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.emb = Embedder(Tokens.EXACT + 1, 16)
        self.conv1 = nn.Conv3d(
            6, 16, kernel_size=(5, 26, 16), stride=(1, 1, 1), bias=False
        )
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 128)
        # the part for actions
        self.fc_action1 = nn.Linear(128, 64)
        self.fc_action2 = nn.Linear(64, output_dims)

        # the part for the value function
        self.fc_value1 = nn.Linear(128, 32)
        self.fc_value2 = nn.Linear(32, 1)
        self.value_output = nn.Linear(128, 1)
        self.advantage_output = nn.Linear(128, output_dims)

    def forward(self, x: torch.LongTensor):
        # B,26
        B = x.shape[0]
        y = self.emb(x)
        y = F.leaky_relu(self.conv1(y))
        # 6,16,1,1
        y = F.leaky_relu(self.fc1(y.squeeze().squeeze()))
        y = F.leaky_relu(self.fc2(y.view(B, -1)))
        # action head
        act = self.fc_action2(F.leaky_relu(self.fc_action1(y)))
        maxa = torch.max(act)
        exp = torch.exp(act - maxa)
        prob = exp / torch.sum(exp)
        m = Categorical(prob)
        action = m.sample()
        action_prob = m.log_prob(action)

        a = self.advantage_output(y)
        v = self.value_output(y)
        v = v.expand_as(a)
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return action, action_prob, prob, q


class Captials(nn.Module):
    def __init__(self, params: dict):
        super(Captials, self).__init__()
        self.seed = torch.manual_seed(params["seed"])
        self.emb = nn.Embedding(3, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(832, params["nA"])

    def forward(self, x: torch.LongTensor):
        B = x.shape[0]
        y = self.emb(x)
        y = F.leaky_relu(self.fc1(y))
        y = F.leaky_relu(self.fc2(y.view(B, -1)))
        return y


class Letters(nn.Module):
    def __init__(self, params):
        super(Letters, self).__init__()
        self.seed = torch.manual_seed(params["seed"])
        self.fc1 = nn.Linear(26, 54)
        self.fc2 = nn.Linear(54, params["nA"])

    def forward(self, x: torch.LongTensor):
        y = F.leaky_relu(self.fc1(x.float()))
        y = F.leaky_relu(self.fc2(y))
        return y
