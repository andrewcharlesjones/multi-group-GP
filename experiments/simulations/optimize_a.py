import autograd.numpy as np
import jax.numpy as jnp
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
from tqdm import tqdm

from multigroupGP import GP, multigroup_rbf_kernel, rbf_kernel

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


n_repeats = 10
p = 1
noise_scale_true = 0.1
n0 = 100
n1 = 100
n_groups = 2
n_params = 5


def separated_gp():

    fitted_params = np.empty((n_repeats, n_params))

    for ii in tqdm(range(n_repeats)):

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
        X_groups = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
        mggp = GP(kernel=multigroup_rbf_kernel, is_mggp=True)

        curr_group_dists = np.ones((n_groups, n_groups))
        np.fill_diagonal(curr_group_dists, 0)
        mggp.fit(
            X,
            Y,
            groups=X_groups,
            group_distances=curr_group_dists,
            verbose=False,
            group_specific_noise_terms=True,
        )
        # assert len(mggp.params) == n_params + 2
        noise_variances = np.exp(mggp.params[1:3])
        output_scale = np.exp(mggp.params[3])
        curr_a = np.exp(mggp.params[4])
        lengthscale = np.exp(mggp.params[5])

        fitted_params[ii, :] = np.array(
            [output_scale, curr_a, lengthscale, noise_variances[0], noise_variances[1]]
        )

    fitted_output_scales, fitted_as, fitted_lengthscales, fitted_noise_variances = (
        fitted_params[:, 0],
        fitted_params[:, 1],
        fitted_params[:, 2],
        fitted_params[:, 3:],
    )

    return fitted_output_scales, fitted_as, fitted_lengthscales, fitted_noise_variances


def union_gp():

    fitted_params = np.empty((n_repeats, n_params))

    for ii in tqdm(range(n_repeats)):

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
        X_groups = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
        mggp = GP(kernel=multigroup_rbf_kernel, is_mggp=True)

        curr_group_dists = np.ones((n_groups, n_groups))
        np.fill_diagonal(curr_group_dists, 0)
        mggp.fit(
            X,
            Y,
            groups=X_groups,
            group_distances=curr_group_dists,
            verbose=False,
            group_specific_noise_terms=True,
        )
        # assert len(mggp.params) == n_params + 2
        noise_variances = np.exp(mggp.params[1:3])
        output_scale = np.exp(mggp.params[3])
        curr_a = np.exp(mggp.params[4])
        lengthscale = np.exp(mggp.params[5])

        fitted_params[ii, :] = np.array(
            [output_scale, curr_a, lengthscale, noise_variances[0], noise_variances[1]]
        )

    fitted_output_scales, fitted_as, fitted_lengthscales, fitted_noise_variances = (
        fitted_params[:, 0],
        fitted_params[:, 1],
        fitted_params[:, 2],
        fitted_params[:, 3:],
    )

    return fitted_output_scales, fitted_as, fitted_lengthscales, fitted_noise_variances


if __name__ == "__main__":

    plt.figure(figsize=(25, 5))

    (
        fitted_output_scales_sep,
        fitted_as_sep,
        fitted_lengthscales_sep,
        fitted_noise_variances_sep,
    ) = separated_gp()
    (
        fitted_output_scales_union,
        fitted_as_union,
        fitted_lengthscales_union,
        fitted_noise_variances_union,
    ) = union_gp()

    plt.subplot(141)
    results_df_output_scales = pd.melt(
        pd.DataFrame(
            {
                "Separated GP": fitted_output_scales_sep,
                "Union GP": fitted_output_scales_union,
            }
        )
    )
    sns.boxplot(data=results_df_output_scales, x="variable", y="value", color="gray")
    plt.axhline(1.0, linestyle="--", color="black")
    plt.xlabel("Data generating model")
    plt.ylabel("Fitted value")
    plt.title(r"$\sigma^2$")
    plt.tight_layout()

    plt.subplot(142)
    results_df_as = pd.melt(
        pd.DataFrame({"Separated GP": fitted_as_sep, "Union GP": fitted_as_union})
    )
    sns.boxplot(data=results_df_as, x="variable", y="value", color="gray")
    plt.xlabel("Data generating model")
    plt.ylabel("Fitted value")
    plt.yscale("log")
    plt.title(r"$a$")
    plt.tight_layout()

    plt.subplot(143)
    results_df_lengthscales = pd.melt(
        pd.DataFrame(
            {
                "Separated GP": fitted_lengthscales_sep,
                "Union GP": fitted_lengthscales_union,
            }
        )
    )
    sns.boxplot(data=results_df_lengthscales, x="variable", y="value", color="gray")
    plt.axhline(1.0, linestyle="--", color="black")
    plt.xlabel("Data generating model")
    plt.ylabel("Fitted value")
    plt.title(r"$b$")
    plt.tight_layout()

    plt.subplot(144)
    # results_df_noise_scales = pd.melt(
    #     pd.DataFrame(
    #         {
    #             "Separated GP,\n group 1": fitted_noise_variances_sep[:, 0],
    #             "Union GP,\n group 1": fitted_noise_variances_union[:, 0],
    #             "Separated GP,\n group 2": fitted_noise_variances_sep[:, 1],
    #             "Union GP,\n group 2": fitted_noise_variances_union[:, 1],
    #         }
    #     )
    # )
    results_df_noise_scales = pd.melt(
        pd.DataFrame(
            {
                "Separated GP": np.concatenate(
                    [fitted_noise_variances_sep[:, 0], fitted_noise_variances_sep[:, 1]]
                ),
                "Union GP": np.concatenate(
                    [
                        fitted_noise_variances_union[:, 0],
                        fitted_noise_variances_union[:, 1],
                    ]
                ),
                "Group": np.concatenate(
                    [np.ones(n_repeats), np.ones(n_repeats) * 2]
                ).astype(int),
            }
        ),
        id_vars="Group",
    )
    sns.boxplot(data=results_df_noise_scales, x="variable", y="value", hue="Group")
    plt.axhline(noise_scale_true ** 2, linestyle="--", color="black")
    plt.xlabel("Data generating model")
    plt.ylabel("Fitted value")
    plt.title(r"$\tau^2$")
    plt.legend(bbox_to_anchor=(1.1, 1.05), borderaxespad=0, title="Group")
    plt.tight_layout()
    plt.savefig("../../plots/parameters_optimized.png")
    plt.show()
    import ipdb

    ipdb.set_trace()
