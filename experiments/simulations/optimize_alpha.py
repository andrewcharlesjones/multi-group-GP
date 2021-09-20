import autograd.numpy as np
import autograd.numpy.random as npr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
import autograd.scipy.stats.multivariate_normal as mvn
from scipy.stats import multivariate_normal as mvnpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from autograd import value_and_grad

sys.path.append("../../models")
from gaussian_process import GP, HGP, MGGP, multigroup_rbf_covariance, rbf_covariance

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


n_repeats = 20
p = 1
noise_scale_true = 0.1
n0 = 20
n1 = 20
n_groups = 2
n_params = 3


def separated_gp():

    # fitted_as = np.empty(n_repeats)
    fitted_params = np.empty((n_repeats, n_params))

    for ii in range(n_repeats):

        ## Generate data
        X0 = np.random.uniform(low=-10, high=10, size=(n0, p))
        X1 = np.random.uniform(low=-10, high=10, size=(n1, p))
        kernel = RBF()
        K_X0X0 = kernel(X0, X0)
        K_X1X1 = kernel(X1, X1)
        Y0 = mvnpy.rvs(np.zeros(n0), K_X0X0) + np.random.normal(
            scale=noise_scale_true, size=n0
        )
        Y1 = mvnpy.rvs(np.zeros(n1), K_X1X1) + np.random.normal(
            scale=noise_scale_true, size=n1
        )

        X0_groups = np.zeros((n0, 1))
        X1_groups = np.ones((n1, 1))
        X = np.concatenate([X0, X1], axis=0)
        Y = np.concatenate([Y0, Y1])

        ############################
        ######### Fit MGGP #########
        ############################
        X0_group_one_hot = np.zeros(n_groups)
        X0_group_one_hot[0] = 1
        X0_groups = np.repeat([X0_group_one_hot], n0, axis=0)
        X1_group_one_hot = np.zeros(n_groups)
        X1_group_one_hot[1] = 1
        X1_groups = np.repeat([X1_group_one_hot], n1, axis=0)
        X_groups = np.concatenate([X0_groups, X1_groups], axis=0)
        mggp = MGGP(kernel=multigroup_rbf_covariance)

        curr_group_dists = np.ones((n_groups, n_groups))
        mggp.fit(X, Y, groups=X_groups, group_distances=curr_group_dists)
        assert len(mggp.params) == n_params + 2
        output_scale = np.exp(mggp.params[2])
        curr_a = np.exp(mggp.params[3])
        lengthscale = np.exp(mggp.params[4])

        fitted_params[ii, :] = np.array([output_scale, curr_a, lengthscale])

    fitted_output_scales, fitted_as, fitted_lengthscales = (
        fitted_params[:, 0],
        fitted_params[:, 1],
        fitted_params[:, 2],
    )

    return fitted_output_scales, fitted_as, fitted_lengthscales


def union_gp():

    fitted_params = np.empty((n_repeats, n_params))

    for ii in range(n_repeats):

        ## Generate data
        X = np.random.uniform(low=-10, high=10, size=(n0 + n1, p))
        kernel = RBF()
        K_XX = kernel(X, X)
        Y = mvnpy.rvs(np.zeros(n0 + n1), K_XX) + np.random.normal(
            scale=noise_scale_true, size=n0 + n1
        )

        ############################
        ######### Fit MGGP #########
        ############################
        X0_group_one_hot = np.zeros(n_groups)
        X0_group_one_hot[0] = 1
        X0_groups = np.repeat([X0_group_one_hot], n0, axis=0)
        X1_group_one_hot = np.zeros(n_groups)
        X1_group_one_hot[1] = 1
        X1_groups = np.repeat([X1_group_one_hot], n1, axis=0)
        X_groups = np.concatenate([X0_groups, X1_groups], axis=0)
        mggp = MGGP(kernel=multigroup_rbf_covariance)

        curr_group_dists = np.ones((n_groups, n_groups))
        mggp.fit(X, Y, groups=X_groups, group_distances=curr_group_dists)
        assert len(mggp.params) == n_params + 2
        output_scale = np.exp(mggp.params[2])
        curr_a = np.exp(mggp.params[3])
        lengthscale = np.exp(mggp.params[4])

        fitted_params[ii, :] = np.array([output_scale, curr_a, lengthscale])

    fitted_output_scales, fitted_as, fitted_lengthscales = (
        fitted_params[:, 0],
        fitted_params[:, 1],
        fitted_params[:, 2],
    )

    return fitted_output_scales, fitted_as, fitted_lengthscales


if __name__ == "__main__":

    plt.figure(figsize=(21, 5))

    fitted_output_scales_sep, fitted_as_sep, fitted_lengthscales_sep = separated_gp()
    fitted_output_scales_union, fitted_as_union, fitted_lengthscales_union = union_gp()

    plt.subplot(131)
    results_df_output_scales = pd.melt(
        pd.DataFrame(
            {
                "Separated GP": fitted_output_scales_sep,
                "Union GP": fitted_output_scales_union,
            }
        )
    )
    sns.boxplot(data=results_df_output_scales, x="variable", y="value")
    plt.axhline(1.0, linestyle="--", color="black")
    plt.xlabel("Model from which data is generated")
    plt.ylabel("Fitted value")
    plt.title(r"$\sigma^2$")
    plt.tight_layout()

    plt.subplot(132)
    results_df_as = pd.melt(
        pd.DataFrame({"Separated GP": fitted_as_sep, "Union GP": fitted_as_union})
    )
    sns.boxplot(data=results_df_as, x="variable", y="value")
    plt.xlabel("Model from which data is generated")
    plt.ylabel("Fitted value")
    plt.yscale("log")
    plt.title(r"$a$")
    plt.tight_layout()

    plt.subplot(133)
    results_df_lengthscales = pd.melt(
        pd.DataFrame(
            {
                "Separated GP": fitted_lengthscales_sep,
                "Union GP": fitted_lengthscales_union,
            }
        )
    )
    sns.boxplot(data=results_df_lengthscales, x="variable", y="value")
    plt.axhline(1.0, linestyle="--", color="black")
    plt.xlabel("Model from which data is generated")
    plt.ylabel("Fitted value")
    plt.title(r"$b$")
    plt.tight_layout()
    plt.savefig("../../plots/parameters_optimized.png")
    plt.show()
    import ipdb

    ipdb.set_trace()
