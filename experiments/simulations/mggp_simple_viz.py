import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
import sys

sys.path.append("../../models")
sys.path.append("../../kernels")
from gaussian_process import GP, HGP, MGGP
from kernels import multigroup_rbf_covariance, rbf_covariance

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
p = 1
xlims = [-10, 10]


def MGGP_experiment():
    ## Generate data from independent GP for each group

    length_scale = 1.0
    alpha_true = 1e0

    ## Generate data
    X0 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n0, p))
    X1 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n1, p))
    X0_group_one_hot = np.zeros(n_groups)
    X0_group_one_hot[0] = 1
    X0_groups = np.repeat([X0_group_one_hot], n0, axis=0)
    X1_group_one_hot = np.zeros(n_groups)
    X1_group_one_hot[1] = 1
    X1_groups = np.repeat([X1_group_one_hot], n1, axis=0)
    X_groups = np.concatenate([X0_groups, X1_groups], axis=0)
    X = np.concatenate([X0, X1], axis=0)

    true_kernel_params = [
        np.log(1.0),  # output variance
        np.log(alpha_true),  # alpha
        np.log(1.0),  # length scale
    ]
    true_group_dists = np.ones((2, 2))
    K_XX = multigroup_rbf_covariance(
        true_kernel_params, X, X, X_groups, X_groups, true_group_dists
    )
    noise = np.random.normal(scale=np.sqrt(noise_variance_true), size=n0 + n1)
    Y = mvn.rvs(np.zeros(n0 + n1), K_XX)  # + noise

    Y0 = Y[:n0]
    Y1 = Y[n0:]

    mggp = MGGP(kernel=multigroup_rbf_covariance)
    curr_group_dists = np.ones((2, 2))
    mggp.fit(X, Y, groups=X_groups, group_distances=curr_group_dists)

    plt.scatter(X, Y)
    preds, _ = mggp.predict(X, groups_test=X_groups)
    plt.scatter(X, preds)
    plt.show()
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    MGGP_experiment()
