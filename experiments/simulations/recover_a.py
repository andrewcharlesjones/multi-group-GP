import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm

from multigroupGP import GP, RBF, MultiGroupRBF

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


n_repeats = 10
p = 1
noise_scale_true = 1e-3
output_scale_true = 1.0
length_scale_true = 1.0
n0 = 100
n1 = 100
alpha_list = alpha_list = [10 ** (x * 1.0) for x in np.arange(-3, 1)]
FIX_TRUE_PARAMS = True
xlims = [-10, 10]
n_groups = 2
n_params = 3


def recover_alpha_experiment():

    fitted_as = np.empty((n_repeats, len(alpha_list)))

    for ii in tqdm(range(n_repeats)):

        for jj, alpha_true in enumerate(alpha_list):

            ## Generate data
            X0 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n0, p))
            X1 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n1, p))
            X0_group_one_hot = np.zeros(n_groups)
            X0_group_one_hot[0] = 1
            X0_groups = np.repeat([X0_group_one_hot], n0, axis=0)
            X1_group_one_hot = np.zeros(n_groups)
            X1_group_one_hot[1] = 1
            X1_groups = np.repeat([X1_group_one_hot], n1, axis=0)
            X_groups = np.concatenate([X0_groups, X1_groups], axis=0).astype(int)
            X = np.concatenate([X0, X1], axis=0)

            true_kernel_params = [
                np.log(output_scale_true),  # output variance
                np.log(alpha_true),         # a
                np.log(length_scale_true),  # length scale
            ]
            true_group_dists = np.ones((2, 2))
            np.fill_diagonal(true_group_dists, 0)
            K_XX = MultiGroupRBF(amplitude=output_scale_true, group_diff_param=alpha_true, lengthscale=length_scale_true)(
                x1=X, groups1=X_groups, group_distances=true_group_dists, log_params=False
            ) + noise_scale_true**2 * np.eye(n0 + n1)
            Y = mvn.rvs(np.zeros(n0 + n1), K_XX)

            ############################
            ######### Fit MGGP #########
            ############################
            X_groups = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
            mggp = GP(kernel=MultiGroupRBF(), is_mggp=True)

            curr_group_dists = np.ones((n_groups, n_groups)) - np.eye(n_groups)
            mggp.fit(X, Y, groups=X_groups, group_distances=curr_group_dists)
            assert len(mggp.params) == n_params + 2
            output_scale = np.exp(mggp.params[2])
            curr_a = np.exp(mggp.params[3])
            lengthscale = np.exp(mggp.params[4])

            fitted_as[ii, jj] = curr_a

    results_df = pd.melt(pd.DataFrame(fitted_as, columns=alpha_list))
    plt.figure(figsize=(5, 5))
    sns.lineplot(data=results_df, x="variable", y="value", err_style="bars")
    plt.yscale("log")
    plt.xscale("log")
    plt.tight_layout()
    plt.xlabel(r"True $a$")
    plt.ylabel(r"Estimated $a$")
    plt.savefig("../../plots/recovering_alpha.png")
    plt.show()
    import ipdb

    ipdb.set_trace()

    return fitted_as


if __name__ == "__main__":

    recover_alpha_experiment()
