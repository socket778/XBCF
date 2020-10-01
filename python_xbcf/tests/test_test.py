import numpy as np
from collections import OrderedDict

# import unittest
import sys
import time

from xbcf import XBCF


def rmse(y1, y2):
    return np.sqrt(np.mean((y1 - y2) ** 2))


def discrete_function(x):
    level = 15 - 20 * (x[:, 0] - 25) ** 2 / 1500
    level = (
        level
        + 15 * np.logical_and(x[:, 1], x[:, 2])
        - 10 * np.logical_or(x[:, 3], x[:, 4])
    )
    level = level * (2 * x[:, 3] - 1)
    return level


def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K] = 1
    np.random.shuffle(arr)
    return arr


n = 500
d = 5
prob = np.random.uniform(0.2, 0.8, d)

x = np.empty([n, d])
x[:, 0] = np.random.normal(25, 10, n)
for h in range(1, d):
    x[:, h] = np.random.binomial(1, prob[h], n)
print(x[0, :])
ftrue = discrete_function(x)
sigma = 0.5 * np.std(ftrue)

# z = rand_bin_array(int(n / 2), n)
z = np.random.choice([0, 1], size=(n,), p=[0.5, 0.5])
z = z.astype(np.int32)
y = ftrue + sigma * np.random.rand(n)

