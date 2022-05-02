from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *
import numpy as np
from ML.utils import reward_over_states
from globals import (
    Embeddings,
    PolicyOutputs,
    DynamicOutputs,
    Tokens,
    Dims,
    index_result_dict,
    alphabet_dict,
)
from torch.distributions import Categorical
from ML.transformer import CTransformer
from abc import ABC, abstractmethod
from operator import itemgetter

from utils import to_tensor, return_rewards


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


LETTER_EMB = nn.Embedding(28, Dims.EMBEDDING_SIZE, padding_idx=0)
RESULT_EMB = nn.Embedding(Tokens.EXACT + 1, Dims.EMBEDDING_SIZE, padding_idx=0)
WORD_EMB = nn.Embedding(12972, Dims.EMBEDDING_SIZE)
ROW_EMB = nn.Embedding(6, Dims.EMBEDDING_SIZE)
COL_EMB = nn.Embedding(5, Dims.EMBEDDING_SIZE)
TURN_EMB = nn.Embedding(6, Dims.EMBEDDING_SIZE)


class AbstractNetwork(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def policy(self, state):
        pass

    @abstractmethod
    def dynamics(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


class Preprocess(nn.Module):
    def __init__(self, config):
        super(Preprocess, self).__init__()
        self.config = config
        self.result_emb = nn.Embedding(
            Tokens.EXACT + 1, Dims.EMBEDDING_SIZE, padding_idx=0
        )
        self.letter_emb = nn.Embedding(28, Dims.EMBEDDING_SIZE, padding_idx=0)
        self.action_emb = nn.Embedding(
            config.action_space + 1, Dims.EMBEDDING_SIZE, padding_idx=0
        )
        self.col_emb = nn.Embedding(5, Dims.EMBEDDING_SIZE)
        self.row_emb = nn.Embedding(6, Dims.EMBEDDING_SIZE)
        self.positional_emb = nn.Embedding(30, Dims.EMBEDDING_SIZE)

    def forward(self, state: torch.tensor):
        assert state.dim() == 4
        B = state.shape[0]
        device = state.get_device()
        if device == -1:
            device = "cpu"
        # print(state)
        res = self.result_emb(state[:, :, :, Embeddings.RESULT])
        letter = self.letter_emb(state[:, :, :, Embeddings.LETTER])
        word = self.action_emb(state[:, :, 0, Embeddings.WORD])
        rows = torch.arange(0, 6).repeat(B, 5).reshape(B, 5, 6).permute(0, 2, 1)
        cols = torch.arange(0, 5).repeat(B, 6).reshape(B, 6, 5)
        row_embs = self.row_emb(rows.to(device))
        col_embs = self.col_emb(cols.to(device))
        positional_embs = row_embs + col_embs
        # 1, 9, 6, 2, 8
        # [1, 6, 5, 8]
        x = res + letter + positional_embs
        y = (word + row_embs[:, :, 0]).unsqueeze(-2)
        # y.shape = (B,6,1,8)
        # x.shape = (B,6,5,8)
        x = torch.cat((x, y), dim=-2)
        # word = self.word_emb(state[:, :, 0, Embeddings.WORD].unsqueeze(-1))
        # x = torch.cat((x, word), dim=-2)
        return x


class StateEncoder(nn.Module):
    def __init__(self, config):
        super(StateEncoder, self).__init__()
        self.process_layer = Preprocess(config)
        self.hidden_state = nn.Linear(240, Dims.HIDDEN_STATE)

    def forward(self, state):
        B = state.shape[0]
        hidden_state = F.leaky_relu(self.process_layer(state))
        hidden_state = F.leaky_relu(self.hidden_state(hidden_state))
        return hidden_state


class TestNet(nn.Module):
    def __init__(
        self,
        config,
        hidden_dims=(Dims.EMBEDDING_SIZE * 5, 256, 508),
        output_dims=(Dims.TRANSFORMER_OUTPUT + Dims.EMBEDDING_SIZE, 256, 256),
    ):
        super(TestNet, self).__init__()
        self.process_input = Preprocess(config)
        self.action_emb = nn.Embedding(config.action_space + 1, 10)
        # self.lstm = nn.LSTM(Dims.TRANSFORMER_INPUT, 64, bidirectional=True)
        self.transformer = CTransformer(
            10,
            heads=5,
            depth=5,
            seq_length=7,
            num_classes=243,
        )
        self.fc_action1 = nn.Linear(253, 243)
        self.fc_action2 = nn.Linear(243, 243)
        self.output = nn.Linear(243, Dims.RESULT_STATE)

    def forward(self, state, action):
        assert state.dim() == 4, f"expect dim of 4 got {state.shape}"
        assert action.dim() == 2, f"expect dim of 2 got {action.shape}"
        B = state.shape[0]
        a = self.action_emb(action)
        if a.dim() == 2:  # for batch training
            a = a.unsqueeze(1)
        # (B,6,5,2)
        x = state.view(B, 6, -1)
        # x = self.process_input(state).view(B, 6, -1)
        x = torch.cat((x, a), dim=1)
        # (B,6,5,8)
        x = self.transformer(x)
        # (B,500)
        return DynamicOutputs(
            F.log_softmax(x, dim=1),
            None,
            None,
        )

    def dynamics(self, state, action):
        return self.forward(state, action)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


def compute_one(state_letters, action_letter, results, device, embedder, col_embs):
    # takes one word and corresponding state. returns 5,6,5 one hot matrix
    # state_letters (6,5)
    # action_letter (5)
    # results (6,5)
    assert state_letters.shape == (6, 5), f"Expected (6,5) {state_letters.shape}"
    assert action_letter.shape == torch.Size([5]), f"Expected (5) {action_letter.shape}"
    assert results.shape == (6, 5), f"Expected (6,5) {results.shape}"
    assert col_embs.shape == (6, 5, 10), f"Expected (6,5,10) {col_embs.shape}"
    res = []
    for i, letter in enumerate(action_letter):
        one_hot = torch.zeros(6, 5).to(device)
        mask = torch.where(state_letters == letter)
        one_hot[mask] = 1
        scaled_results = embedder((one_hot * results).long()) + col_embs
        res.append(scaled_results)
    res = torch.stack(res)
    assert res.shape == (5, 6, 5, 10), f"Expected (5,6,5,10) {res.shape}"
    return res


def compute_batch(
    state_letters, action_letters, result_batch, batch_len, device, embedder, col_embs
):
    attention = [
        compute_one(
            state_letters[i],
            action_letters[i],
            result_batch[i],
            device,
            embedder,
            col_embs,
        )
        for i in range(batch_len)
    ]
    return torch.stack(attention)


class StateActionTransition(nn.Module):
    def __init__(self, config):
        super(StateActionTransition, self).__init__()
        self.process_input = Preprocess(config)
        self.result_emb = nn.Embedding(Tokens.EXACT + 1, 10, padding_idx=0)
        self.letter_emb = nn.Embedding(28, Dims.EMBEDDING_SIZE, padding_idx=0)
        self.col_emb = nn.Embedding(5, 10)
        self.config = config
        self.action_emb = nn.Embedding(12973, 15)
        # self.history_transformer = CTransformer(
        #     10,
        #     heads=5,
        #     depth=5,
        #     seq_length=7,
        #     num_classes=Dims.RESULT_STATE,
        # )

        # self.result_transformer = CTransformer(
        #     300,
        #     heads=10,
        #     depth=5,
        #     seq_length=5,
        #     num_classes=Dims.RESULT_STATE,
        # )
        # self.result_compute = mlp(300,[128,128],32)
        # self.output_layer = mlp(288,[256,256],Dims.RESULT_STATE)
        self.transformer = CTransformer(
            15, heads=15, depth=10, seq_length=7, num_classes=Dims.RESULT_STATE
        )

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def forward(self, state, action):
        assert state.dim() == 4, f"expect dim of 4 got {state.shape}"
        assert action.dim() == 2, f"expect dim of 2 got {action.shape}"
        B = state.shape[0]

        device = state.get_device()
        if device == -1:
            device = "cpu"
        # state_letters = state[:,:,:,Embeddings.LETTER] # (B,6,5)
        # state_results = state[:,:,:,Embeddings.RESULT] # (B,6,5)
        # # previous_actions = state[:,:,0,Embeddings.WORD] # (B,1)
        # action_letters = torch.tensor(np.stack([np.array([alphabet_dict[letter] for letter in self.config.index_to_word[a.item()]]) for a in action])).to(device) # (B,5)
        # cols = torch.arange(0, 5).repeat(6).reshape(6, 5).to(device)

        # col_embs = self.col_emb(cols) # (6,5,emb)
        # result_attention = compute_batch(state_letters,action_letters,state_results,B,device,self.result_emb,col_embs) # (B,5,6,5,emb) ([512, 5, 512, 6, 5, 10]) ([1, 5, 1, 6, 5, 10])
        # print('result_attention',result_attention.shape)
        # assert result_attention.shape == (B,5,6,5,10), f"Expected (B,5,6,5,10) {result_attention.shape}"
        # computed_result = self.result_compute(result_attention.view(B,5,-1)).view(B,-1) # (B,160)
        # print('computed_result.shape',computed_result.shape)
        # result_embs = self.result_emb(result_attention.long())
        # result_embs = result_embs + col_embs # (B,5,6,5,emb)
        # result_embs = result_embs.view(B,5,-1)
        # letter_embs = self.letter_emb(action_letters)
        # state_input = state[:,:,:,:2].contiguous().view(B, 6, -1) # (B,6,10)
        x = self.process_input(state)
        a = self.action_emb(action)
        if a.dim() == 2:
            a = a.unsqueeze(1)
        # # print(result_embs.shape,state_input.shape,action_letters.shape,a.shape)
        x = torch.cat((state.view(B, 6, -1), a), dim=1)
        x = self.transformer(x)  # (B,128)
        # print('x.shape',x.shape)
        # x = torch.cat((x,computed_result),dim=-1)
        # x = self.output_layer(x)
        # y = torch.cat((result_embs,letter_embs),dim=-1)
        # # print('xy',x.shape,y.shape)
        # x = self.history_transformer(x)
        # y = self.result_transformer(y)
        # x = torch.cat((x,y),dim=-1)
        # x = self.fc_action2(F.leaky_relu(self.fc_action1(x)))
        # print(result_embs.get_device(),action_letters.get_device(),state.get_device())
        # x = torch.cat((result_embs.view(B,-1),state.view(B,-1)),dim=-1)

        # (B,6,5,3)
        # prev_actions = self.action_emb(state[:, :, 0, Embeddings.WORD])
        # norm_prev = prev_actions / torch.sqrt(prev_actions.pow(2))
        # # (B,6,emb_size)
        # norm_action = a / torch.sqrt(a.pow(2))
        # # (B,emb_size)
        # x = state[:, :, :, : Embeddings.WORD]
        # # B,6,10
        # similarity = torch.bmm(norm_prev, norm_action.view(B, 10, 1))
        # # B,6,1
        # x = x + similarity
        # B,6,10
        # x = torch.cat((x, a), dim=1)
        # B, 7, 40
        # B, 243
        m = Categorical(logits=x)
        turns = torch.count_nonzero(state, dim=1)[:, 0, 0].view(-1, 1)
        bools = torch.where(turns >= 5, 1, 0)
        rewards = reward_over_states(bools)
        return DynamicOutputs(
            F.log_softmax(x, dim=1),
            m.probs,
            rewards,
        )


class ZeroPolicy(nn.Module):
    def __init__(self, config):
        super(ZeroPolicy, self).__init__()
        self.seed = torch.manual_seed(1234)
        self.process_input = Preprocess(config)
        self.transformer = CTransformer(
            48,
            heads=8,
            depth=5,
            seq_length=6,
            num_classes=Dims.OUTPUT,
        )
        # the part for actions
        self.fc_action1 = nn.Linear(Dims.TRANSFORMER_OUTPUT, 256)
        self.fc_action2 = nn.Linear(256, config.action_space)

        # the part for the value function
        self.value_output = nn.Linear(Dims.TRANSFORMER_OUTPUT, 1)
        self.advantage_output = nn.Linear(256, config.action_space)

    def forward(self, state: torch.LongTensor):
        # B,26
        B = state.shape[0]
        x = self.process_input(state).view(B, 6, -1)
        # [B, 6, 40]
        x = self.transformer(x)
        # [B, 500]
        # action head
        act = self.fc_action2(F.leaky_relu(self.fc_action1(x)))
        maxa = torch.max(act)
        exp = torch.exp(act - maxa)
        probs = exp / torch.sum(exp)
        m = Categorical(probs)
        action = m.sample()
        v = self.value_output(x)
        return PolicyOutputs(action, F.log_softmax(m.logits, dim=1), m.probs, v)


class MuZeroNet(AbstractNetwork):
    def __init__(self, config):
        super(MuZeroNet, self).__init__()
        self.config = config
        if config.train_on_gpu:
            self._policy = torch.nn.DataParallel(ZeroPolicy(config))
            self._representation = torch.nn.DataParallel(StateEncoder(config))
            self._dynamics = torch.nn.DataParallel(StateActionTransition(config))
        else:
            self._policy = ZeroPolicy(config)
            self._representation = StateEncoder(config)
            self._dynamics = StateActionTransition(config)

    def representation(self, state):
        return self._representation(state)

    def dynamics(self, state, action) -> DynamicOutputs:
        return self._dynamics(state, action)

    def policy(self, state) -> PolicyOutputs:
        return self._policy(state)


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


class Threshold(nn.Module):
    def __init__(self, params: dict):
        super(Threshold, self).__init__()
        self.process_input = Preprocess()
        self.seed = torch.manual_seed(params["seed"])
        ff = [nn.Linear(60, 16)]
        for _ in range(params["n_layers"]):
            ff.append(nn.Linear(16, 16))
        ff.append(nn.Linear(16, 2))
        self.ff = nn.Sequential(*ff)

    def forward(self, x: torch.LongTensor):
        B = x.shape[0]
        y = x.view(B, -1)
        for ff in self.ff:
            y = F.leaky_relu(ff(y))
        return y
