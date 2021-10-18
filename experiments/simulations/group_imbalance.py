import numpy as np
import numpy.random as npr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_normal as mvnpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from multigroupGP import GP, MultiGroupRBF, RBF, HGPKernel

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


n_repeats = 20
p = 1
noise_variance_true = 0.1
n_per_group = np.array([10, 50, 50])
n_groups = 3
frac_train = 0.5
FIX_TRUE_PARAMS = False
n0_list = [5, 10, 30, 50]
xlims = [-10, 10]

g12_dist = 1e-5
g13_dist = 1e1
g23_dist = 1e1
a_true = 1e0
group_dist_mat = np.array(
    [
        [0.0, g12_dist, g13_dist],
        [g12_dist, 0.0, g23_dist],
        [g13_dist, g23_dist, 0.0],
    ]
)


def generate_mggp_data(n_groups, n_per_group, p=1):

    n = np.sum(n_per_group)

    kernel_params = [np.log(1.0)] * 3
    kernel_params[1] = np.log(a_true)
    mggp_kernel = MultiGroupRBF(amplitude=1.0, group_diff_param=1.0, lengthscale=1.0)
    kernel = lambda X1, X2, groups1, groups2: mggp_kernel(
        x1=X1,
        x2=X2,
        groups1=groups1,
        groups2=groups2,
        group_distances=group_dist_mat,
    )

    ## Generate data
    groups_list = []
    for gg in range(n_groups):
        curr_groups = np.ones(n_per_group[gg]) * gg

        groups_list.append(curr_groups)
    X_groups = np.concatenate(groups_list, axis=0).astype(int)

    X = np.random.uniform(low=xlims[0], high=xlims[1], size=(n, p))

    curr_K_XX = kernel(X, X, X_groups, X_groups) + noise_variance_true * np.eye(n)
    Y = mvn.rvs(np.zeros(n), curr_K_XX)

    return X, Y, X_groups


def experiment():

    errors_mggp = np.empty((n_repeats, len(n0_list)))
    errors_separated_gp = np.empty((n_repeats, len(n0_list)))
    errors_union_gp = np.empty((n_repeats, len(n0_list)))
    errors_hgp = np.empty((n_repeats, len(n0_list)))

    for ii in tqdm(range(n_repeats)):

        for jj, n0 in enumerate(n0_list):
            ## Generate data from MGGP

            n_per_group[0] = n0
            X, Y, X_groups = generate_mggp_data(
                n_groups=n_groups, n_per_group=n_per_group
            )
            (
                X_train,
                X_test,
                Y_train,
                Y_test,
                X_groups_train,
                X_groups_test,
            ) = train_test_split(
                X, Y, X_groups, test_size=1 - frac_train, random_state=42
            )

            ############################
            ######### Fit MGGP #########
            ############################

            mggp = GP(kernel=MultiGroupRBF(), is_mggp=True)
            mggp.fit(X_train, Y_train, X_groups_train, group_dist_mat)
            preds_mean = mggp.predict(X_test, X_groups_test)

            group0_idx = X_groups_test == 0
            curr_error = np.mean((Y_test[group0_idx] - preds_mean[group0_idx]) ** 2)
            errors_mggp[ii, jj] = curr_error

            ############################
            ##### Fit separated GP #####
            ############################

            sum_error_sep_gp = 0
            for groupnum in range(n_groups):
                curr_X_train = X_train[X_groups_train == groupnum]
                curr_Y_train = Y_train[X_groups_train == groupnum]
                curr_X_test = X_test[X_groups_test == groupnum]
                curr_Y_test = Y_test[X_groups_test == groupnum]

                sep_gp = GP(kernel=RBF())
                sep_gp.fit(curr_X_train, curr_Y_train)
                preds_mean = sep_gp.predict(curr_X_test)
                curr_error_sep_gp = np.sum((curr_Y_test - preds_mean) ** 2)

                if groupnum == 0:
                    sum_error_sep_gp += curr_error_sep_gp

            # errors_separated_gp[ii, jj] = sum_error_sep_gp / Y_test.shape[0]
            errors_separated_gp[ii, jj] = sum_error_sep_gp / sum(group0_idx.astype(int))

            ############################
            ####### Fit union GP #######
            ############################

            union_gp = GP(kernel=RBF())
            union_gp.fit(X_train, Y_train)
            preds_mean = union_gp.predict(X_test)
            # curr_error = np.mean((Y_test - preds_mean) ** 2)
            curr_error = np.mean((Y_test[group0_idx] - preds_mean[group0_idx]) ** 2)
            errors_union_gp[ii, jj] = curr_error

            ############################
            ######### Fit HGP ##########
            ############################
            hgp_kernel = HGPKernel(
                within_group_kernel=RBF(), between_group_kernel=RBF()
            )
            hgp = GP(kernel=hgp_kernel, is_hgp=True)
            hgp.fit(X_train, Y_train, X_groups_train)
            preds_mean = hgp.predict(X_test, X_groups_test)
            curr_error = np.mean((Y_test[group0_idx] - preds_mean[group0_idx]) ** 2)
            errors_hgp[ii, jj] = curr_error

    errors_mggp_df = pd.melt(pd.DataFrame(errors_mggp, columns=n0_list))
    errors_mggp_df["model"] = ["MGGP"] * errors_mggp_df.shape[0]
    errors_separated_gp_df = pd.melt(pd.DataFrame(errors_separated_gp, columns=n0_list))
    errors_separated_gp_df["model"] = ["Separated GP"] * errors_separated_gp_df.shape[0]
    errors_union_gp_df = pd.melt(pd.DataFrame(errors_union_gp, columns=n0_list))
    errors_union_gp_df["model"] = ["Union GP"] * errors_union_gp_df.shape[0]
    errors_hgp_df = pd.melt(pd.DataFrame(errors_hgp, columns=n0_list))
    errors_hgp_df["model"] = ["HGP"] * errors_hgp_df.shape[0]

    results_df = pd.concat(
        [errors_mggp_df, errors_separated_gp_df, errors_union_gp_df, errors_hgp_df],
        axis=0,
    )

    return results_df, X, Y, X_groups


if __name__ == "__main__":

    results_df, X, Y, X_groups = experiment()

    plt.figure(figsize=(7, 5))

    g = sns.lineplot(data=results_df, x="variable", y="value", hue="model")
    g.legend_.set_title(None)
    plt.ylabel(r"Prediction MSE, group $1$")
    plt.xlabel(r"$n$, group 1")
    plt.tight_layout()
    plt.savefig("../../plots/three_group_simulated_prediction.png")
    plt.show()

    import ipdb

    ipdb.set_trace()
