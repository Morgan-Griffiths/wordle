from tkinter import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *
import numpy as np
from globals import Embeddings, Outputs, State, Tokens, Dims, index_result_dict
from torch.distributions import Categorical

from utils import to_tensor


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


STATE_CONV = nn.Conv3d(6, 16, kernel_size=(1, 26, 16), stride=(1, 1, 1), bias=False)
LETTER_EMB = nn.Embedding(28, Dims.EMBEDDING_SIZE, padding_idx=0)
RESULT_EMB = nn.Embedding(Tokens.EXACT + 1, Dims.EMBEDDING_SIZE, padding_idx=0)
WORD_EMB = nn.Embedding(12972, Dims.EMBEDDING_SIZE)
ROW_EMB = nn.Embedding(6, Dims.EMBEDDING_SIZE)
COL_EMB = nn.Embedding(5, Dims.EMBEDDING_SIZE)
TURN_EMB = nn.Embedding(6, Dims.EMBEDDING_SIZE)


class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()
        self.result_emb = RESULT_EMB
        self.letter_emb = LETTER_EMB
        self.col_emb = COL_EMB
        self.row_emb = ROW_EMB
        self.positional_emb = nn.Embedding(30, Dims.EMBEDDING_SIZE)

    def forward(self, state):
        B = state.shape[0]
        rows = torch.arange(0, 6).repeat(B, 5).reshape(B, 5, 6).permute(0, 2, 1)
        cols = torch.arange(0, 5).repeat(B, 6).reshape(B, 6, 5)
        row_embs = self.row_emb(rows)
        col_embs = self.col_emb(cols)
        positional_embs = row_embs + col_embs
        res = self.result_emb(state[:, :, :, Embeddings.RESULT])
        letter = self.letter_emb(state[:, :, :, Embeddings.LETTER])
        x = res + letter + positional_embs
        # word = self.word_emb(state[:, :, 0, Embeddings.WORD].unsqueeze(-1))
        # x = torch.cat((x, word), dim=-2)
        return x


class StateEncoder(nn.Module):
    def __init__(self):
        super(StateEncoder, self).__init__()
        self.process_layer = Preprocess()
        self.hidden_state = nn.Linear(240, Dims.HIDDEN_STATE)

    def forward(self, state):
        B = state.shape[0]
        hidden_state = F.leaky_relu(self.process_layer(state))
        hidden_state = F.leaky_relu(self.hidden_state(hidden_state))
        return hidden_state


class StateActionTransition(nn.Module):
    def __init__(
        self,
        hidden_dims=(Dims.EMBEDDING_SIZE * 5, 256, 256),
        output_dims=(256 * 6 + Dims.EMBEDDING_SIZE, 256, 256),
    ):
        super(StateActionTransition, self).__init__()
        self.process_layer = Preprocess()
        self.action_emb = WORD_EMB
        self.activation_fc = F.leaky_relu
        self.action_position = ROW_EMB
        self.ff_layers = [
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        ]
        self.output_layers = [
            nn.Linear(output_dims[i], output_dims[i + 1])
            for i in range(1, len(output_dims) - 1)
        ]
        self.output_layer = nn.Linear(hidden_dims[-1] * 6 + Dims.EMBEDDING_SIZE, 256)
        self.result = nn.Linear(hidden_dims[-1], Dims.RESULT_STATE)
        self.average_reward = nn.Linear(hidden_dims[-1], 1)
        self.reward = nn.Linear(hidden_dims[-1], Dims.RESULT_STATE)

    def forward(self, state, action):
        B = state.shape[0]
        x = self.process_layer(state).view(B, 6, -1)
        turn = torch.where(state[:, :, 0] == 0)[1].min()
        r_boolean = 1 if turn < 5 else -1
        action_position = self.action_position(turn)
        a = self.action_emb(action) + action_position
        for hidden_layer in self.ff_layers:
            x = self.activation_fc(hidden_layer(x))

        s = torch.cat((x.view(B, -1), a), dim=-1)
        s = F.leaky_relu(self.output_layer(s))
        result_logits = self.result(s)
        s = s * r_boolean
        for hidden_layer in self.output_layers:
            s = self.activation_fc(hidden_layer(s))
        # rewards
        rewards = self.reward(s)
        # average_reward = self.average_reward(s)
        # average_reward = average_reward.expand_as(rewards_adv)
        # rewards = (
        #     average_reward
        #     + rewards_adv
        #     - rewards_adv.mean(1, keepdim=True).expand_as(rewards_adv)
        # )

        # results
        m = Categorical(logits=result_logits)
        result = m.sample()
        return (
            index_result_dict[result[0].item()],
            result_logits,
            rewards[:, result[0].item()],
            rewards,
        )


class ZeroPolicy(nn.Module):
    def __init__(self, config, output_dims):
        super(ZeroPolicy, self).__init__()
        self.seed = torch.manual_seed(config.seed)
        self.process_input = Preprocess()
        self.fc1 = nn.Linear(240, 128)
        self.fc2 = nn.Linear(128, 128)
        # the part for actions
        self.fc_action1 = nn.Linear(128, 64)
        self.fc_action2 = nn.Linear(64, output_dims)

        # the part for the value function
        self.value_output = nn.Linear(128, 1)
        self.advantage_output = nn.Linear(128, output_dims)

    def forward(self, x: torch.LongTensor):
        # B,26
        B = x.shape[0]
        y = self.process_input(x)
        # 6,16,1,1
        y = F.leaky_relu(self.fc1(y.view(B, -1)))
        y = F.leaky_relu(self.fc2(y))
        # action head
        act = self.fc_action2(F.leaky_relu(self.fc_action1(y)))
        maxa = torch.max(act)
        exp = torch.exp(act - maxa)
        probs = exp / torch.sum(exp)
        m = Categorical(probs)
        action = m.sample()
        v = self.value_output(y)
        return action, probs, v


class MuZeroNet(nn.Module):
    def __init__(self, config, output_dims):
        super(MuZeroNet, self).__init__()
        self._policy = ZeroPolicy(config, output_dims)
        self._representation = StateEncoder()
        self._dynamics = StateActionTransition()

    def representation(self, state):
        return self._representation(state)

    def dynamics(self, state, action):
        return self._dynamics(state, action)

    def policy(self, state):
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
        self.process_input = Preprocess()
        self.fc1 = nn.Linear(Dims.PROCESSED, 128)
        self.fc2 = nn.Linear(128, 128)

        # the part for the value function
        self.value_output = nn.Linear(128, 1)
        self.advantage_output = nn.Linear(128, output_dims)
        self.noise = GaussianNoise()

    def forward(self, x: torch.LongTensor):
        # B,26
        B = x.shape[0]
        y = self.process_input(x)
        # 6,16,1,1
        y = F.leaky_relu(self.fc1(y.view(B, -1)))
        y = F.leaky_relu(self.fc2(y))
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
            Outputs.VALUES: torch.tanh(q),
        }


class Policy(nn.Module):
    def __init__(self, config, params):
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(config.seed)
        self.process_input = Preprocess()
        self.fc1 = nn.Linear(Dims.PROCESSED, 128)
        self.fc2 = nn.Linear(128, 128)
        # the part for actions
        self.fc_action1 = nn.Linear(128, 64)
        self.fc_action2 = nn.Linear(64, params["nA"])

        # the part for the value function
        self.fc_value1 = nn.Linear(128, 32)
        self.fc_value2 = nn.Linear(32, 1)
        self.value_output = nn.Linear(128, 1)
        self.advantage_output = nn.Linear(128, params["nA"])

    def forward(self, x: torch.LongTensor):
        # B,26
        B = x.shape[0]
        y = self.process_input(x)
        # 6,16,1,1
        y = F.leaky_relu(self.fc1(y.view(B, -1)))
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


class LSTM_letters(nn.Module):
    def __init__(self, seed, nA, hidden_dims=(128, 128)):
        super(LSTM_letters, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_dims = hidden_dims
        self.nA = nA
        self.device = "cpu"

        self.result_emb = RESULT_EMB
        self.letter_emb = LETTER_EMB
        self.lstm = nn.LSTM(16, 256)
        self.input_layer = nn.Linear(16, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(self.hidden_dims)):
            hidden_layer = nn.Linear(hidden_dims[i - 1], hidden_dims[i])
            self.hidden_layers.append(hidden_layer)
        self.actor_output = nn.Linear(hidden_dims[-1], nA)
        self.critic_output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.LongTensor, action=None):
        # B,26
        B = x.shape[0]
        letters = self.letter_emb(x[:, :, 0])
        result = self.result_emb(x[:, :, 1])
        x = torch.cat((letters, result))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        # action head
        logits = self.actor_output(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic state value
        v = self.critic_output(x)
        return action, log_prob, dist.entropy(), v


class PPO(nn.Module):
    def __init__(self, seed, nA, hidden_dims=(128, 128)):
        super(PPO, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_dims = hidden_dims
        self.nA = nA
        self.device = "cpu"
        self.process_input = Preprocess()
        self.input_layer = nn.Linear(Dims.PROCESSED, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(self.hidden_dims)):
            hidden_layer = nn.Linear(hidden_dims[i - 1], hidden_dims[i])
            self.hidden_layers.append(hidden_layer)
        self.actor_output = nn.Linear(hidden_dims[-1], nA)
        self.critic_output = nn.Linear(hidden_dims[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.critic_output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action=None):
        B = state.shape[0]
        x = self.process_input(state)
        x = F.relu(self.input_layer(x.view(B, -1)))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        # state -> action
        logits = self.actor_output(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic state value
        v = self.critic_output(x)
        return action, log_prob, dist.probs, dist.entropy(), v

    # Return the action along with the probability of the action. For weighting the reward garnered by the action.
    def act(self, state, action=None):
        x = state
        if not isinstance(state, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)  # device = self.device,
            x = x.unsqueeze(0)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        logits = self.actor_output(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class Reinforce(nn.Module):
    def __init__(self, params: dict):
        super(Reinforce, self).__init__()
        self.seed = torch.manual_seed(params["seed"])
        self.emb = RESULT_EMB
        self.conv1 = STATE_CONV
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 128)
        # the part for actions
        self.fc_action1 = nn.Linear(128, 64)
        self.fc_action2 = nn.Linear(64, params["nA"])

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
        probs = exp / torch.sum(exp)
        m = Categorical(probs)
        action = m.sample()
        action_prob = m.log_prob(action)
        return action, action_prob, probs


### Experimental Networks


class WordleTest(nn.Module):
    def __init__(self, params: dict):
        super(WordleTest, self).__init__()
        self.seed = torch.manual_seed(params["seed"])
        self.process_input = Preprocess()
        self.fc1 = nn.Linear(240, 128)
        self.fc2 = nn.Linear(128, params["nA"])

    def forward(self, x: torch.LongTensor):
        B = x.shape[0]
        y = self.process_input(x)
        y = F.leaky_relu(self.fc1(y.view(B, -1)))
        y = F.leaky_relu(self.fc2(y.view(B, -1)))
        return y


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
