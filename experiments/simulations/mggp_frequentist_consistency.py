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


sys.path.append("../../models")
sys.path.append("../../kernels")
from gaussian_process import GP, HGP, MGGP
from kernels import multigroup_rbf_covariance, rbf_covariance

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


n_repeats = 20
p = 1
noise_scale_true = 0
output_scale_true = 1.0
length_scale_true = 1.0
FIX_TRUE_PARAMS = True
xlims = [-10, 10]
n_groups = 2
n_params = 3
alpha_true = 1e0
# n_per_group_list = np.linspace(5, 101, 3).astype(int)
n_per_group_list = [2, 5, 10]


def recover_alpha_experiment():

    fitted_as = np.empty((n_repeats, len(n_per_group_list)))

    for ii in tqdm(range(n_repeats)):

        for jj, n_per_group in enumerate(n_per_group_list):

            ## Generate data
            X0 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n_per_group, p))
            X1 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n_per_group, p))
            X0_group_one_hot = np.zeros(n_groups)
            X0_group_one_hot[0] = 1
            X0_groups = np.repeat([X0_group_one_hot], n_per_group, axis=0)
            X1_group_one_hot = np.zeros(n_groups)
            X1_group_one_hot[1] = 1
            X1_groups = np.repeat([X1_group_one_hot], n_per_group, axis=0)
            X_groups = np.concatenate([X0_groups, X1_groups], axis=0)
            X = np.concatenate([X0, X1], axis=0)

            true_kernel_params = [
                np.log(output_scale_true),  # output variance
                np.log(alpha_true),  # alpha
                np.log(length_scale_true),  # length scale
            ]
            true_group_dists = np.ones((2, 2))
            K_XX = multigroup_rbf_covariance(
                true_kernel_params, X, X, X_groups, X_groups, true_group_dists
            )
            noise = np.random.normal(
                scale=noise_scale_true, size=n_per_group * n_groups
            )
            Y = mvn.rvs(np.zeros(n_per_group * n_groups), K_XX) + noise

            ############################
            ######### Fit MGGP #########
            ############################
            X0_group_one_hot = np.zeros(n_groups)
            X0_group_one_hot[0] = 1
            X0_groups = np.repeat([X0_group_one_hot], n_per_group, axis=0)
            X1_group_one_hot = np.zeros(n_groups)
            X1_group_one_hot[1] = 1
            X1_groups = np.repeat([X1_group_one_hot], n_per_group, axis=0)
            X_groups = np.concatenate([X0_groups, X1_groups], axis=0)
            mggp = MGGP(kernel=multigroup_rbf_covariance)

            curr_group_dists = np.ones((n_groups, n_groups))
            mggp.fit(X, Y, groups=X_groups, group_distances=curr_group_dists)
            assert len(mggp.params) == n_params + 2
            output_scale = np.exp(mggp.params[2])
            curr_a = np.exp(mggp.params[3])
            lengthscale = np.exp(mggp.params[4])

            fitted_as[ii, jj] = curr_a
            # import ipdb; ipdb.set_trace()

    results_df = pd.melt(pd.DataFrame(fitted_as, columns=n_per_group_list))
    plt.figure(figsize=(7, 5))
    # sns.lineplot(data=results_df, x="variable", y="value", err_style="bars")
    sns.boxplot(data=results_df, x="variable", y="value", color="Gray")
    # plt.xscale("log")
    plt.axhline(alpha_true, linestyle="--", color="red", label=r"True $a$")
    plt.legend()
    plt.yscale("log")
    # plt.xscale("log")
    plt.tight_layout()
    plt.xlabel(r"$n$")
    plt.ylabel(r"Estimated $a$")
    plt.savefig("../../plots/mggp_consistency_frequentist.png")
    plt.show()
    import ipdb

    ipdb.set_trace()

    return fitted_as


if __name__ == "__main__":

    recover_alpha_experiment()
