import os
import sys
import time
import numpy as np
from numpy.random import choice
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import genfromtxt

from xbart import XBART
from xbcausalforest import XBCF

np.set_printoptions(threshold=sys.maxsize)


def rmse(y1, y2):
    return np.sqrt(np.mean((y1 - y2) ** 2))


# 1. Load data (y, mu, tau, z, X)
path = os.getcwd()  # works only in case it's launched from examples folder
fileloc = path + "/newdf.csv"  # need to fix it
print(fileloc)

my_data = genfromtxt(fileloc, delimiter=",")

y = my_data[:, 0]
mu = my_data[:, 1]
tau = my_data[:, 2]
z = my_data[:, 3]
X = my_data[:, 4:9]
z = z.astype(np.int32)

d_t = X.shape[1]
n = X.shape[0]


# 2. treatment effect estimation

# scale response variable
meany = np.mean(y)
y = y - np.mean(y)
sdy = np.std(y)
y = y / sdy

# fit propensity scores using XBART
start = time.time()
xbt = XBART(num_trees=100, num_sweeps=40, burnin=15)
xb_fit = xbt.fit_predict(X, z, X)
xb_fit = xb_fit.reshape((n, 1))

end = time.time()
print("second elapsed XBART for pihat: ", end - start)

# append the fitted propensity scores to the original matrix
X1 = np.hstack((xb_fit, X))
d_p = X1.shape[1]

# train-test split
n_test = 100

# train
X_train = X[0 : n - n_test, :]
X1_train = X1[0 : n - n_test, :]
tau_train = tau[0 : n - n_test]
y_train = y[0 : n - n_test]
z_train = z[0 : n - n_test]

# test
X_test = X[n - n_test : n, :]
tau_test = tau[n - n_test : n]


# XBCF parameters
sweeps = 40
burn = 15
p_cat = 2
trees_pr = 30
trees_trt = 10

print("XBCF fit")

model = XBCF(
    num_sweeps=sweeps,
    burnin=burn,
    max_depth=250,
    num_trees_pr=trees_pr,
    num_trees_trt=trees_trt,
    mtry_pr=d_p,
    mtry_trt=d_t,
    num_cutpoints=100,
    Nmin=1,
    p_categorical_pr=p_cat,
    p_categorical_trt=p_cat,
    tau_pr=0.6 * np.var(y) / trees_pr,
    tau_trt=0.1 * np.var(y) / trees_trt,
    no_split_penality="auto",
    parallel=True,
)

start = time.time()
obj_train = model.fit(X_train, X1_train, y_train, z_train)
end = time.time()
print("second elapsed XBCF: ", end - start)

b = obj_train.b.transpose()
a = obj_train.a.transpose()

# In-sample fit
thats = sdy * obj_train.tauhats * (b[1] - b[0])
thats_mean = np.mean(thats[:, (burn) : (sweeps - 1)], axis=1)


print("CATE rmse train: ", rmse(tau_train, thats_mean))
plt.scatter(tau_train, thats_mean)
plt.xlabel("tau")
plt.ylabel("tauhats")
plt.show()

# Out-of-sample fit
print("==== OOS fit example ====")

tauhats_test = model.predict(X_test)

thats_test = sdy * tauhats_test * (b[1] - b[0])
thats_test_mean = np.mean(thats_test[:, (burn) : (sweeps - 1)], axis=1)


print("CATE rmse test: ", rmse(tau_test, thats_test_mean))
plt.scatter(tau_test, thats_test_mean)
plt.xlabel("tau_test")
plt.ylabel("tauhats_test")
plt.show()

