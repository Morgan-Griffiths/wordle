import torch
from einops import rearrange
import functools
import numpy as np
import pandas as pd
from collections import Counter
import torch.nn as nn


class Node:
    def __init__(
        self,
        Y,
        X,
        min_samples_split=None,
        max_depth=None,
        depth=None,
        node_type=None,
        rule=None,
    ):
        self.Y = Y
        self.X = X
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5

        # Default current depth of node
        self.depth = depth if depth else 0
        # Extracting all the features
        self.features = list(self.X.columns)

        # Type of node
        self.node_type = node_type if node_type else "root"

        # Rule for spliting
        self.rule = rule if rule else ""

        # Calculating the counts of Y in the node
        self.counts = Counter(Y)

        # Getting the GINI impurity based on the Y distribution
        self.gini_impurity = self.get_GINI()

        # Sorting the counts and saving the final prediction of the node
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        # Getting the last item
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]

        # Saving to object attribute. This node will predict the class with the most frequent class
        self.yhat = yhat

        # Saving the number of observations in the node
        self.n = len(Y)

        # Initiating the left and right nodes as empty nodes
        self.left = None
        self.right = None

        # Default values for splits
        self.best_feature = None
        self.best_value = None

    @staticmethod
    def GINI_impurity(y1_count: int, y2_count: int) -> float:
        """
        Given the observations of a binary class calculate the GINI impurity
        """
        # Ensuring the correct types
        if y1_count is None:
            y1_count = 0

        if y2_count is None:
            y2_count = 0

        # Getting the total observations
        n = y1_count + y2_count

        # If n is 0 then we return the lowest possible gini impurity
        if n == 0:
            return 0.0

        # Getting the probability to see each of the classes
        p1 = y1_count / n
        p2 = y2_count / n

        # Calculating GINI
        gini = 1 - (p1 ** 2 + p2 ** 2)

        # Returning the gini impurity
        return gini

    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list.
        """
        return np.convolve(x, np.ones(window), "valid") / window

    def get_GINI(self):
        """
        Function to calculate the GINI impurity of a node
        """
        # Getting the 0 and 1 counts
        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)

        # Getting the GINI impurity
        return self.GINI_impurity(y1_count, y2_count)


# Decision Tree
# Cost functions - Gini index and entropy
# Gini = 1 - SUM (P_i)^2. P_i = probability of having that class or value. Ranges from 0 to 0.5 (split evenly)


def gini_impurity(y):
    p = y.value_counts() / y.shape[0]
    gini = 1 - np.sum(p ** 2)
    return gini


# Entropy E(S) = SUM -p_i * log_2 (p_i)


def entropy(y):
    a = y.value_counts() / y.shape[0]
    return np.sum(-a * np.log2(a + 1e-9))


# class NN:
#     def __init__(self, x, y):
#         self.input = x
#         self.weights1 = np.random.rand(self.input.shape[1], 4)
#         self.weights2 = np.random.rand(4, 1)
#         self.y = y
#         self.output = np.zeros(y.shape)

#     def forward(self, x):
#         self.layer1 = sigmoid(np.dot(self.input, self.weights1))
#         self.output = sigmoid(np.dot(self.layer1, self.weights2))

#     def backward(self):
#         ...


# def loss(x, y):
#     return np.sum((x - y) ** 2)


# def sigmoid(x):
#     return 1 / 1 + np.exp(-x)


# inputs = np.array([1, 5, 10, 2])


# with open("/usr/share/dict/words", "r") as f:
#     words = f.read()


# def timeit(func):
#     import time

#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         tic = time.perf_counter()
#         print(f"running function {func.__name__}")
#         res = func(*args, **kwargs)
#         print(f"Elapsed {time.perf_counter() - tic:4f} secs")
#         return res

#     return wrapper


# @timeit
# def waste_time(num):
#     for _ in range(num):
#         sum([i for i in range(10000)])


# @timeit
# def say_whee():
#     print("whee")


# @timeit
# def greet(name):
#     return f"Hello {name}"


# print(waste_time(2))

# states = torch.tensor(
#     [
#         [
#             [[1, 3], [2, 1], [1, 2], [3, 1], [1, 1]],
#             [[1, 3], [2, 1], [1, 2], [3, 1], [1, 1]],
#             [[1, 3], [1, 3], [12, 1], [9, 1], [9, 1]],
#             [[1, 3], [1, 3], [8, 2], [5, 1], [4, 1]],
#             [[1, 3], [1, 3], [18, 3], [20, 1], [9, 1]],
#             [[1, 3], [1, 3], [12, 1], [9, 1], [9, 1]],
#         ],
#         [
#             [[1, 3], [2, 1], [1, 2], [3, 1], [1, 1]],
#             [[1, 3], [2, 1], [1, 2], [3, 1], [1, 1]],
#             [[1, 3], [1, 3], [12, 1], [9, 1], [9, 1]],
#             [[1, 3], [1, 3], [8, 2], [5, 1], [4, 1]],
#             [[1, 3], [1, 3], [18, 3], [20, 1], [9, 1]],
#             [[1, 3], [1, 3], [12, 1], [9, 1], [9, 1]],
#         ],
#     ]
# )

# print(torch.where(states[:, :, :, 0] == 0))
# turn = torch.where(states[:, :, :, 0] == 0)[1]
# r_boolean = torch.where(turn < 5, 1, -1)
# print(turn)
# print(r_boolean)


x = torch.ones(1680)
y = torch.ones(1680)
x = x.view(5, 6, 7, 8)
y = y.view(5, 6, 7, 8)
b, h, t, e = x.shape
print(x.shape)
x_rearrange = rearrange(x, "b h t e -> (b h) t e")
print("orign y", y.shape)
print("trans y", y.transpose(1, 2).contiguous().shape)
einsum_y = y.transpose(1, 2).contiguous().view(b * h, t, e)
print("x_rearrange", x_rearrange.shape)
print("einsum_y", einsum_y.shape)
print("x", x.shape)
print("y", y.shape)


a = torch.randn(10, 20, 30)  # b -> 10, i -> 20, k -> 30
c = torch.randn(10, 50, 30)  # b -> 10, j -> 50, k -> 30
y1 = torch.einsum("b i k, b j k -> b i j", a, c)  # shape [10, 20, 50]
print(y1.shape)

out_bmm = torch.bmm(a, c.permute(0, 2, 1))
print(out_bmm.shape)


qkv = torch.rand(2, 128, 3 * 512)  # 2, 128, 1536 b,t,(n k)
q, k, v = tuple(rearrange(qkv, "b t (n k) -> n b t k", n=3))
print(q.shape, k.shape, v.shape)


# Transformer
# batch, tokens, dim
# Q = XW
# K = XW
# V = XW
dim = 512
tokens = 128
x = torch.rand(32, 128, dim)
to_qvk = nn.Linear(dim, dim * 3, bias=False)
qkv = to_qvk(x)
print("qkv", qkv.shape)
q, v, k = tuple(rearrange(qkv, "b t (n k) -> n b t k", n=3))
print(q.shape, k.shape, v.shape)

# calc dot product, apply mask, compute softmax in d dimension
# softmax(QK^T/sqrt(d_k))

scaled_dot_product = torch.einsum("b i d, b j d -> b i j", q, k) * torch.sqrt(torch.tensor(dim))

print("scaled_dot_product", scaled_dot_product.shape)
attention = torch.softmax(scaled_dot_product, dim=-1)
print("attention", attention.shape)
# multiply by V
# attention = softmax(QK^T/sqrt(d_k)) * V

# self_attention = torch.einsum("b i j, b ", attention, v)
