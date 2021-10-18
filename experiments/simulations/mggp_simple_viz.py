import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
import sys
from multigroupGP import GP, MultiGroupRBF, RBF

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

n_repeats = 5
alpha_list = alpha_list = [np.power(10, x * 1.0) for x in np.arange(-5, 3)]
noise_variance_true = 0.1
n_groups = 2
n0 = 100
n1 = 100
n = n0 + n1
p = 1
xlims = [-10, 10]


def MGGP_experiment():
    ## Generate data from independent GP for each group

    length_scale = 1.0
    alpha_true = 1e0

    ## Generate data
    X0 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n0, p))
    X1 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n1, p))
    X_groups = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
    X = np.concatenate([X0, X1], axis=0)

    true_group_dists = np.ones((2, 2)) - np.eye(2)
    K_XX = MultiGroupRBF(amplitude=1.0, group_diff_param=1.0, lengthscale=1.0)(
        x1=X, groups1=X_groups, group_distances=true_group_dists, log_params=False
    ) + noise_variance_true * np.eye(n)
    Y = mvn.rvs(np.zeros(n0 + n1), K_XX)

    Y0 = Y[:n0]
    Y1 = Y[n0:]

    mggp = GP(kernel=MultiGroupRBF(), is_mggp=True)
    curr_group_dists = np.ones((2, 2)) - np.eye(2)
    mggp.fit(X, Y, groups=X_groups, group_distances=curr_group_dists)

    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y, color="black")

    ntest = 200
    Xtest_onegroup = np.linspace(xlims[0], xlims[1], ntest)[:, None]
    Xtest = np.concatenate([Xtest_onegroup, Xtest_onegroup], axis=0)
    Xtest_groups = np.concatenate([np.zeros(ntest), np.ones(ntest)]).astype(int)
    preds = mggp.predict(Xtest, groups_test=Xtest_groups)
    plt.plot(Xtest[:ntest], preds[:ntest], color="blue")
    plt.plot(Xtest[ntest:], preds[ntest:], color="red")
    plt.show()
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    MGGP_experiment()
