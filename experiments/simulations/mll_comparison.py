import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.gaussian_process.kernels import RBF as RBF_sklearn
import sys

# sys.path.append("../../models")
# sys.path.append("../../kernels")
# from gaussian_process import GP, HGP, MGGP
# from kernels import multigroup_rbf_covariance, rbf_covariance
from multigroupGP import GP, MultiGroupRBF, RBF, HGPKernel

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

n_repeats = 20
alpha_list = alpha_list = [np.power(10, x * 1.0) for x in np.arange(-5, 3)]
noise_variance_true = 0.1
n_groups = 2
n0 = 100
n1 = 100
n = n0 + n1
p = 1
xlims = [-10, 10]


def separate_GP_experiment():
    ## Generate data from independent GP for each group

    ll_mggp_results = np.empty((n_repeats, len(alpha_list)))
    ll_sep_gp_results = np.empty(n_repeats)
    ll_union_gp_results = np.empty(n_repeats)
    for ii in range(n_repeats):

        ## Generate data
        X0 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n0, p))
        X1 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n1, p))
        kernel = RBF_sklearn()
        K_X0X0 = kernel(X0, X0) + noise_variance_true * np.eye(n0)
        K_X1X1 = kernel(X1, X1) + noise_variance_true * np.eye(n1)
        Y0 = mvn.rvs(np.zeros(n0), K_X0X0)
        Y1 = mvn.rvs(np.zeros(n1), K_X1X1)

        ############################
        ######### Fit SGP ##########
        ############################
        sep_gp = GP(kernel=RBF())
        curr_params = {
            "params": np.array(
                [
                    0.0,  # mean
                    np.log(noise_variance_true),  # noise variance
                    np.log(1.0),  # output variance
                    np.log(1.0),  # length scale
                ]
            )
        }
        ll_sep_group0 = sep_gp.log_marginal_likelihood(curr_params, X0, Y0)
        ll_sep_group1 = sep_gp.log_marginal_likelihood(curr_params, X1, Y1)
        ll_sep_gp_results[ii] = ll_sep_group0 + ll_sep_group1

        ############################
        ######### Fit UGP ##########
        ############################
        X = np.concatenate([X0, X1], axis=0)
        Y = np.concatenate([Y0, Y1])
        union_gp = GP(kernel=RBF())
        ll_union = union_gp.log_marginal_likelihood(curr_params, X, Y)
        ll_union_gp_results[ii] = ll_union

        ############################
        ######### Fit MGGP #########
        ############################
        X_groups = np.concatenate([np.zeros(n0), np.ones(n1)], axis=0).astype(int)
        X = np.concatenate([X0, X1], axis=0)
        Y = np.concatenate([Y0, Y1])
        for jj, alpha in enumerate(alpha_list):
            mggp = GP(kernel=MultiGroupRBF(), is_mggp=True)
            curr_params = {
                "params": np.array(
                    [
                        0.0,  # mean
                        np.log(noise_variance_true),  # noise variance
                        np.log(1.0),  # output variance
                        np.log(alpha),  # alpha
                        np.log(1.0),  # length scale
                    ]
                )
            }
            curr_group_dists = np.ones((2, 2)) - np.eye(2)
            ll_mggp = mggp.log_marginal_likelihood(
                curr_params, X, Y, groups=X_groups, group_distances=curr_group_dists
            )
            ll_mggp_results[ii, jj] = ll_mggp

    ## Plot MGGP results
    results_df = pd.melt(pd.DataFrame(ll_mggp_results, columns=alpha_list))
    sns.lineplot(data=results_df, x="variable", y="value", label="Multi-Group", ci="sd")

    ## Plot sep GP results
    gp_results_df = pd.melt(
        pd.DataFrame(
            np.vstack([ll_sep_gp_results, ll_sep_gp_results]).T,
            columns=[alpha_list[0], alpha_list[-1]],
        )
    )
    sns.lineplot(
        data=gp_results_df,
        x="variable",
        y="value",
        label="Separate",
        color="red",
        ci="sd",
    )

    ## Plot union GP results
    gp_results_df = pd.melt(
        pd.DataFrame(
            np.vstack([ll_union_gp_results, ll_union_gp_results]).T,
            columns=[alpha_list[0], alpha_list[-1]],
        )
    )
    sns.lineplot(
        data=gp_results_df,
        x="variable",
        y="value",
        label="Union",
        color="green",
        ci="sd",
    )

    plt.xscale("log")
    plt.xlabel(r"$a$")
    plt.ylabel(r"$\log p(Y)$")
    plt.legend(loc="lower right")
    plt.tight_layout()


def union_GP_experiment():
    ## Generate data from independent GP for each group

    ll_mggp_results = np.empty((n_repeats, len(alpha_list)))
    ll_sep_gp_results = np.empty(n_repeats)
    ll_union_gp_results = np.empty(n_repeats)
    for ii in range(n_repeats):

        ## Generate data
        X0 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n0, p))
        X1 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n1, p))
        X = np.concatenate([X0, X1], axis=0)
        kernel = RBF_sklearn()
        K_XX = kernel(X, X) + noise_variance_true * np.eye(n)
        Y = mvn.rvs(np.zeros(n0 + n1), K_XX)
        Y0, Y1 = Y[:n0], Y[n0:]

        Y = np.concatenate([Y0, Y1])

        ############################
        ######### Fit SGP ##########
        ############################
        sep_gp = GP(kernel=RBF())
        curr_params = {
            "params": np.array(
                [
                    0.0,  # mean
                    np.log(noise_variance_true),  # noise variance
                    np.log(1.0),  # output variance
                    np.log(1.0),  # length scale
                ]
            )
        }
        ll_sep_group0 = sep_gp.log_marginal_likelihood(curr_params, X0, Y0)
        ll_sep_group1 = sep_gp.log_marginal_likelihood(curr_params, X1, Y1)
        ll_sep_gp_results[ii] = ll_sep_group0 + ll_sep_group1

        ############################
        ######### Fit UGP ##########
        ############################
        X = np.concatenate([X0, X1], axis=0)
        Y = np.concatenate([Y0, Y1])
        union_gp = GP(kernel=RBF())
        ll_union = union_gp.log_marginal_likelihood(curr_params, X, Y)
        ll_union_gp_results[ii] = ll_union

        ############################
        ######### Fit MGGP #########
        ############################
        X_groups = np.concatenate([np.zeros(n0), np.ones(n1)], axis=0).astype(int)
        X = np.concatenate([X0, X1], axis=0)
        Y = np.concatenate([Y0, Y1])
        for jj, alpha in enumerate(alpha_list):
            mggp = GP(kernel=MultiGroupRBF(), is_mggp=True)
            curr_params = {
                "params": np.array(
                    [
                        0.0,  # mean
                        np.log(noise_variance_true),  # noise variance
                        np.log(1.0),  # output variance
                        np.log(alpha),  # alpha
                        np.log(1.0),  # length scale
                    ]
                )
            }
            curr_group_dists = np.ones((2, 2)) - np.eye(2)
            ll_mggp = mggp.log_marginal_likelihood(
                curr_params, X, Y, groups=X_groups, group_distances=curr_group_dists
            )
            ll_mggp_results[ii, jj] = ll_mggp

    ## Plot MGGP results
    results_df = pd.melt(pd.DataFrame(ll_mggp_results, columns=alpha_list))
    sns.lineplot(data=results_df, x="variable", y="value", label="Multi-Group", ci="sd")

    ## Plot sep GP results
    gp_results_df = pd.melt(
        pd.DataFrame(
            np.vstack([ll_sep_gp_results, ll_sep_gp_results]).T,
            columns=[alpha_list[0], alpha_list[-1]],
        )
    )
    sns.lineplot(
        data=gp_results_df,
        x="variable",
        y="value",
        label="Separate",
        color="red",
        ci="sd",
    )

    ## Plot union GP results
    gp_results_df = pd.melt(
        pd.DataFrame(
            np.vstack([ll_union_gp_results, ll_union_gp_results]).T,
            columns=[alpha_list[0], alpha_list[-1]],
        )
    )
    sns.lineplot(
        data=gp_results_df,
        x="variable",
        y="value",
        label="Union",
        color="green",
        ci="sd",
    )

    plt.xscale("log")
    plt.xlabel(r"$a$")
    plt.ylabel(r"$\log p(Y)$")
    plt.legend(loc="lower right")
    plt.tight_layout()


def HGP_experiment():
    ## Generate data from independent GP for each group

    length_scale_shared = 1.0
    length_scale_group0 = 1.0
    length_scale_group1 = 1.0

    alpha_list = [np.power(10, x * 1.0) for x in np.arange(-5, 5)]
    ll_mggp_results = np.empty((n_repeats, len(alpha_list)))
    ll_hgp_results = np.empty(n_repeats)
    for ii in range(n_repeats):

        ## Generate data
        X0 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n0, p))
        X1 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n1, p))
        X = np.concatenate([X0, X1], axis=0)
        kernel_shared = RBF_sklearn(length_scale=1.0)
        kernel_group0 = RBF_sklearn(length_scale=length_scale_group0)
        kernel_group1 = RBF_sklearn(length_scale=length_scale_group1)

        K_XX = kernel_shared(X, X)
        K_XX[:n0, :n0] += kernel_group0(X0, X0)
        K_XX[n0:, n0:] += kernel_group1(X1, X1)
        Y = mvn.rvs(np.zeros(n0 + n1), K_XX + noise_variance_true * np.eye(n))

        ############################
        ######### Fit HGP ##########
        ############################
        X_groups = np.concatenate([np.zeros(n0), np.ones(n1)], axis=0).astype(int)
        X = np.concatenate([X0, X1], axis=0)
        hgp_kernel = HGPKernel(within_group_kernel=RBF(), between_group_kernel=RBF())
        hgp = GP(kernel=hgp_kernel, is_hgp=True)
        curr_params = {
            "params": np.array(
                [
                    0.0,  # mean
                    np.log(noise_variance_true),  # noise variance
                    np.log(1.0),  # output variance
                    np.log(1.0),  # length scale
                    np.log(1.0),  # output variance
                    np.log(1.0),  # length scale
                ]
            )
        }
        ll_hgp = hgp.log_marginal_likelihood(curr_params, X, Y, groups=X_groups)
        ll_hgp_results[ii] = ll_hgp

        ############################
        ######### Fit MGGP #########
        ############################
        for jj, alpha in enumerate(alpha_list):
            mggp = GP(kernel=MultiGroupRBF(), is_mggp=True)
            curr_params = {
                "params": np.array(
                    [
                        0.0,  # mean
                        np.log(noise_variance_true),  # noise variance
                        np.log(1.0),  # output variance
                        np.log(alpha),  # alpha
                        np.log(1.0),  # length scale
                    ]
                )
            }
            curr_group_dists = np.ones((2, 2)) - np.eye(2)
            ll_mggp = mggp.log_marginal_likelihood(
                curr_params, X, Y, groups=X_groups, group_distances=curr_group_dists
            )
            ll_mggp_results[ii, jj] = ll_mggp

    # plt.figure(figsize=(7, 5))
    results_df = pd.melt(pd.DataFrame(ll_mggp_results, columns=alpha_list))
    sns.lineplot(data=results_df, x="variable", y="value", label="Multi-Group", ci="sd")

    gp_results_df = pd.melt(
        pd.DataFrame(
            np.vstack([ll_hgp_results, ll_hgp_results]).T,
            columns=[alpha_list[0], alpha_list[-1]],
        )
    )
    sns.lineplot(
        data=gp_results_df,
        x="variable",
        y="value",
        label="Hierarchical",
        color="orange",
        ci="sd",
    )
    plt.xscale("log")
    plt.xlabel(r"$a$")
    plt.ylabel(r"$\log p(Y)$")
    plt.legend(loc="lower right")
    plt.tight_layout()


def MGGP_experiment():
    ## Generate data from independent GP for each group

    length_scale = 1.0
    alpha_true = 1e0

    ll_mggp_results = np.empty((n_repeats, len(alpha_list)))
    ll_sep_gp_results = np.empty(n_repeats)
    ll_union_gp_results = np.empty(n_repeats)
    ll_hgp_results = np.empty(n_repeats)
    for ii in range(n_repeats):

        ## Generate data
        X0 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n0, p))
        X1 = np.random.uniform(low=xlims[0], high=xlims[1], size=(n1, p))
        X_groups = np.concatenate([np.zeros(n0), np.ones(n1)], axis=0).astype(int)
        X = np.concatenate([X0, X1], axis=0)

        true_kernel_params = [
            np.log(1.0),  # output variance
            np.log(alpha_true),  # alpha
            np.log(1.0),  # length scale
        ]
        true_group_dists = np.ones((2, 2)) - np.eye(2)
        K_XX = MultiGroupRBF(
            amplitude=1.0, group_diff_param=alpha_true, lengthscale=1.0
        )(
            x1=X, groups1=X_groups, group_distances=true_group_dists, log_params=False
        ) + noise_variance_true * np.eye(
            n
        )
        Y = mvn.rvs(np.zeros(n0 + n1), K_XX)  # + noise

        Y0 = Y[:n0]
        Y1 = Y[n0:]

        ############################
        ######### Fit SGP ##########
        ############################
        sep_gp = GP(kernel=RBF())
        curr_params = {
            "params": np.array(
                [
                    0.0,  # mean
                    np.log(noise_variance_true),  # noise variance
                    np.log(1.0),  # output variance
                    np.log(1.0),  # length scale
                ]
            )
        }
        ll_sep_group0 = sep_gp.log_marginal_likelihood(curr_params, X0, Y0)
        ll_sep_group1 = sep_gp.log_marginal_likelihood(curr_params, X1, Y1)
        ll_sep_gp_results[ii] = ll_sep_group0 + ll_sep_group1

        ############################
        ######### Fit UGP ##########
        ############################
        X = np.concatenate([X0, X1], axis=0)
        Y = np.concatenate([Y0, Y1])
        union_gp = GP(kernel=RBF())
        ll_union = union_gp.log_marginal_likelihood(curr_params, X, Y)
        ll_union_gp_results[ii] = ll_union

        ############################
        ######### Fit HGP ##########
        ############################
        hgp_kernel = HGPKernel(within_group_kernel=RBF(), between_group_kernel=RBF())
        hgp = GP(kernel=hgp_kernel, is_hgp=True)
        curr_params = {
            "params": np.array(
                [
                    0.0,  # mean
                    np.log(noise_variance_true),  # noise variance
                    np.log(1.0),  # output variance
                    np.log(1.0),  # length scale
                    np.log(1.0),  # output variance
                    np.log(1.0),  # length scale
                ]
            )
        }
        ll_hgp = hgp.log_marginal_likelihood(curr_params, X, Y, groups=X_groups)
        ll_hgp_results[ii] = ll_hgp

        ############################
        ######### Fit MGGP #########
        ############################
        for jj, alpha in enumerate(alpha_list):
            mggp = GP(kernel=MultiGroupRBF(), is_mggp=True)
            curr_params = {
                "params": np.array(
                    [
                        0.0,  # mean
                        np.log(noise_variance_true),  # noise variance
                        np.log(1.0),  # output variance
                        np.log(alpha),  # alpha
                        np.log(1.0),  # length scale
                    ]
                )
            }
            curr_group_dists = np.ones((2, 2)) - np.eye(2)
            ll_mggp = mggp.log_marginal_likelihood(
                curr_params, X, Y, groups=X_groups, group_distances=curr_group_dists
            )
            ll_mggp_results[ii, jj] = ll_mggp

    # plt.figure(figsize=(7, 5))
    results_df = pd.melt(pd.DataFrame(ll_mggp_results, columns=alpha_list))
    sns.lineplot(data=results_df, x="variable", y="value", label="Multi-Group", ci="sd")

    gp_results_df = pd.melt(
        pd.DataFrame(
            np.vstack([ll_sep_gp_results, ll_sep_gp_results]).T,
            columns=[alpha_list[0], alpha_list[-1]],
        )
    )
    sns.lineplot(
        data=gp_results_df,
        x="variable",
        y="value",
        label="Separate",
        color="red",
        ci="sd",
    )

    gp_results_df = pd.melt(
        pd.DataFrame(
            np.vstack([ll_union_gp_results, ll_union_gp_results]).T,
            columns=[alpha_list[0], alpha_list[-1]],
        )
    )
    sns.lineplot(
        data=gp_results_df,
        x="variable",
        y="value",
        label="Union",
        color="green",
        ci="sd",
    )

    gp_results_df = pd.melt(
        pd.DataFrame(
            np.vstack([ll_hgp_results, ll_hgp_results]).T,
            columns=[alpha_list[0], alpha_list[-1]],
        )
    )
    sns.lineplot(
        data=gp_results_df,
        x="variable",
        y="value",
        label="Hierarchical",
        color="orange",
        ci="sd",
    )

    plt.axvline(alpha_true, linestyle="--", alpha=0.3, color="black")

    # import ipdb; ipdb.set_trace()

    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel(r"$a$")
    plt.ylabel(r"$\log p(Y)$")
    plt.legend(loc="lower right")
    plt.tight_layout()
    # plt.savefig("../plots/mggp_comparison.png")
    # plt.show()

    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    plt.figure(figsize=(28, 7))
    plt.subplot(141)
    separate_GP_experiment()
    plt.title(r"Data generated from: $\textbf{Separate}$")
    plt.subplot(142)
    union_GP_experiment()
    plt.title(r"Data generated from: $\textbf{Union}$")
    plt.subplot(143)
    HGP_experiment()
    plt.title(r"Data generated from: $\textbf{Hierarchical}$")
    plt.subplot(144)
    MGGP_experiment()
    plt.title(r"Data generated from: $\textbf{Multi-Group}$")
    plt.savefig("../../plots/simulation_gp_comparison.png")
    plt.show()

    # MGGP_experiment()
