import numpy as np
from numpy.random import choice
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import xbart

# import unittest
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


n = 2000
d = 50

x = np.empty([n, d])
for i in range(0, d - 2):
    x[:, i] = np.random.normal(0, 1, n)
x[:, d - 2] = np.random.choice(2, n, p=[0.5, 0.5])
x[:, d - 1] = np.random.choice(3, n, p=[0.3, 0.4, 0.3])

print(x[1, :])

# options
linear = True
mu = mu(x, linear, n)
homogeneous = False
tau = tau(x, homogeneous, n)

# define the propensity score function
pi = np.zeros(n)
for i in range(n):
    pi[i] = 0.8 * norm.pdf(3 * mu[i] / np.std(mu) - 0.5 * x[i, 0], 0, 1) + 0.05
    +0.1 * np.random.uniform(1, 0, 1)

# generate treatment assignment scheme
z = np.random.binomial(1, pi, n)
z = z.astype(np.int32)

# generate response variable
Ey = mu + tau * z
sig = 0.5 * np.std(Ey)
y = Ey + sig * np.random.normal(0, 1, n)
"""

from numpy import genfromtxt

path = os.getcwd()
fileloc = path + "/tests/newdf.csv"
print(fileloc)

my_data = genfromtxt(fileloc, delimiter=",")

# print(type(my_data))
# print(my_data[0])
y = my_data[:, 0]
mu = my_data[:, 1]
tau = my_data[:, 2]
z = my_data[:, 3]
x = my_data[:, 4:9]
z = z.astype(np.int32)
"""
d_t = x.shape[1]
n = x.shape[0]
# print(my_data[0])
# print(x[0])

# 2. treatment effect estimation

# scale response variable
meany = np.mean(y)
y = y - np.mean(y)
sdy = np.std(y)
y = y / sdy
"""
from xbart import XBART

start = time.time()

xbt = XBART(num_trees=100, num_sweeps=40, burnin=15)
xb_fit = xbt.fit_predict(x, z, x)
# xbart_yhat_matrix = xbt.predict(x_test)  # Return n X num_sweeps matrix
# y_hat = xbart_yhat_matrix[:, 15:].mean(axis=1)  # Use mean a prediction estimate
xb_fit = xb_fit.reshape((n, 1))

end = time.time()
print("second elapsed XBART for pihat: ", end - start)

x1 = np.hstack((xb_fit, x))
"""
# x1 = x
pi = pi.reshape((n, 1))
x1 = np.hstack((pi, x))
d_p = x1.shape[1]

# XBCF parameters
sweeps = 40
burn = 15
p_cat = 2

# print("d_t: ", d_t)
# print("d_p: ", d_p)
# print("x1 rows: ", x1.shape[0])

print("fit it!")

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
# print(type(z[0]))
# params = model.get_params()
# print(params)

start = time.time()
obj = model.fit(x, x1, y, z, p_cat)
end = time.time()
print("second elapsed XBCF: ", end - start)

# print(type(obj.tauhats))
# print("No. of dimensions: ", obj.tauhats.ndim)
# print("Shape of array: ", obj.tauhats.shape)
# print("Size of array: ", obj.tauhats.size)
# print("sdy: ", sdy)
b = obj.b.transpose()
a = obj.a.transpose()
# print(obj.tauhats[0:2, 15:39])
thats = sdy * obj.tauhats * (b[1] - b[0])
thats_mean = np.mean(thats[:, (burn) : (sweeps - 1)], axis=1)
yhats = obj.muhats * a + obj.tauhats * (b[1] - b[0])
yhats_mean = np.mean(yhats[:, (burn) : (sweeps - 1)], axis=1)
# print(yhats_mean)
# print(thats_mean)
# print(obj.b)
# print(obj.a)
print(rmse(yhats_mean, y))
print(rmse(tau, thats_mean))
plt.scatter(tau, thats_mean)
plt.xlabel("tau")
plt.ylabel("tauhats")
plt.show()

