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
n_groups = 3
ns = [20] * n_groups
# n0 = 20
# n1 = 20
n = np.sum(ns)
p = 1
xlims = [-10, 10]
ntest = 200


def MGGP_experiment():    
    ## Generate data
    X = np.concatenate(
        [np.random.uniform(low=xlims[0], high=xlims[1], size=(ns[ii], p)) for ii in range(n_groups)]
    )
    X_groups = np.concatenate([
        np.ones(ns[ii]) * ii for ii in range(n_groups)
    ]).astype(int)

    true_group_dists = np.ones((n_groups, n_groups)) - np.eye(n_groups)
    K_XX = MultiGroupRBF(amplitude=1.0, group_diff_param=2.0, lengthscale=2.0)(
        x1=X, groups1=X_groups, group_distances=true_group_dists, log_params=False
    ) + noise_variance_true * np.eye(n)
    Y = mvn.rvs(np.zeros(n), K_XX)


    mggp = GP(kernel=MultiGroupRBF(), is_mggp=True)
    curr_group_dists = np.ones((n_groups, n_groups)) - np.eye(n_groups)
    mggp.fit(X, Y, groups=X_groups, group_distances=curr_group_dists)

    colors = ["blue", "red", "green"]
    Xtest_onegroup = np.linspace(xlims[0], xlims[1], ntest)[:, None]
    Xtest = np.concatenate([Xtest_onegroup] * n_groups, axis=0)
    Xtest_groups = np.concatenate([
        np.ones(ntest) * ii for ii in range(n_groups)
    ]).astype(int)
    preds, preds_cov = mggp.predict(Xtest, groups_test=Xtest_groups, return_cov=True)
    preds_stddev_diag = np.sqrt(np.diagonal(preds_cov))

    plt.figure(figsize=(10, 5))
    for gg in range(n_groups):
        plt.scatter(X[X_groups == gg], Y[X_groups == gg], color=colors[gg], label="Group {}".format(gg + 1), s=100)
        plt.plot(Xtest[Xtest_groups == gg], preds[Xtest_groups == gg], color=colors[gg])
        plt.fill_between(
            Xtest[Xtest_groups == gg].squeeze(),
            preds[Xtest_groups == gg] - 2 * preds_stddev_diag[Xtest_groups == gg],
            preds[Xtest_groups == gg] + 2 * preds_stddev_diag[Xtest_groups == gg],
            color=colors[gg],
            alpha=0.3,
        )
    plt.legend()
    plt.show()
    import ipdb

    ipdb.set_trace()

if __name__ == "__main__":
    MGGP_experiment()
