import numpy as np
from numpy.random import choice
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

import sys
import time
from scipy.stats import norm

from xbcausalforest import XBCF

np.set_printoptions(threshold=sys.maxsize)


def rmse(y1, y2):
    return np.sqrt(np.mean((y1 - y2) ** 2))


def mu(x, lin, len):
    lev = [2, -1, -4]
    result = np.zeros(len)
    if lin:
        for i in range(len):
            result[i] = 1 + lev[x[i, 49].astype(np.int32)] + x[i, 0] * x[i, 2]
    else:
        for i in range(len):
            result[i] = -6 + lev[x[i, 49].astype(np.int32)] + 6 * np.abs(x[i, 2] - 1)
    return result


def tau(x, h, len):
    if h:
        result = [3] * len
    else:
        result = 1 + 2 * x[:, 1] * x[:, 48]
    return result


# 1. Generate the data
n = 20000
d = 50

X = np.empty([n, d])
for i in range(0, d - 2):
    X[:, i] = np.random.normal(0, 1, n)
X[:, d - 2] = np.random.choice(2, n, p=[0.5, 0.5])
X[:, d - 1] = np.random.choice(3, n, p=[0.3, 0.4, 0.3])


# options
linear = True
homogeneous = False


# generate mu and tau
mu = mu(X, linear, n)
tau = tau(X, homogeneous, n)


# define the propensity score function
pi = np.zeros(n)
for i in range(n):
    pi[i] = 0.8 * norm.pdf(3 * mu[i] / np.std(mu) - 0.5 * X[i, 0], 0, 1) + 0.05
    +0.1 * np.random.uniform(1, 0, 1)


# generate treatment assignment scheme
z = np.random.binomial(1, pi, n)
z = z.astype(np.int32)


# generate response variable
Ey = mu + tau * z
sig = 0.5 * np.std(Ey)
y = Ey + sig * np.random.normal(0, 1, n)


d_t = X.shape[1]
n = X.shape[0]


# 2. treatment effect estimation

# scale response variable
meany = np.mean(y)
y = y - np.mean(y)
sdy = np.std(y)
y = y / sdy

# if we don't know pi (propensity scores), we would estimate it here
"""
from xbart import XBART

start = time.time()

xbt = XBART(num_trees=100, num_sweeps=40, burnin=15)
xb_fit = xbt.fit_predict(x, z, x)
# xbart_yhat_matrix = xbt.predict(x_test)  # Return n X num_sweeps matrix
# y_hat = xbart_yhat_matrix[:, 15:].mean(axis=1)  # Use mean a prediction estimate
pi = xb_fit.reshape((n, 1))

end = time.time()
print("second elapsed XBART for pihat: ", end - start)
"""

# append propensity scores to the original data to enhance prognostic term estimation
pi = pi.reshape((n, 1))
X1 = np.hstack((pi, X))
d_p = X1.shape[1]

# XBCF parameters
sweeps = 40
burn = 15
p_cat = 2

print("XBCF fit")

model = XBCF(
    num_sweeps=sweeps,
    burnin=burn,
    max_depth=250,
    num_trees_pr=30,
    num_trees_trt=10,
    mtry_pr=d_p,
    mtry_trt=d_t,
    num_cutpoints=20,
    Nmin=1,
    p_categorical_pr=p_cat,
    p_categorical_trt=p_cat,
    tau_pr=0.6 * np.var(y) / 30,
    tau_trt=0.1 * np.var(y) / 10,
    no_split_penality="auto",
    parallel=True,
)


start = time.time()
obj = model.fit(X, X1, y, z)
end = time.time()
print("second elapsed XBCF: ", end - start)

b = obj.b.transpose()
a = obj.a.transpose()

thats = sdy * obj.tauhats * (b[1] - b[0])
thats_mean = np.mean(thats[:, (burn) : (sweeps - 1)], axis=1)
yhats = obj.muhats * a + obj.tauhats * (b[1] - b[0])
yhats_mean = np.mean(yhats[:, (burn) : (sweeps - 1)], axis=1)


print("CATE rmse: ", rmse(tau, thats_mean))
plt.scatter(tau, thats_mean)
plt.xlabel("tau")
plt.ylabel("tauhats")
plt.show()

