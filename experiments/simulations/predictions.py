import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

from multigroupGP import GP, multigroup_rbf_kernel, rbf_kernel, hgp_kernel

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

KERNEL_TYPE = "rbf"

if KERNEL_TYPE == "matern":
    gp_kernel = matern12_covariance
    mggp_kernel = multigroup_matern12_covariance
    n_mgg_kernel_params = 4
elif KERNEL_TYPE == "rbf":
    gp_kernel = rbf_kernel
    mggp_kernel = multigroup_rbf_kernel
    n_mgg_kernel_params = 3


n_repeats = 3
noise_variance_true = 0.1
n_groups = 3
# n_per_group = 10
n_per_group = 100
n = n_per_group * n_groups
p = 1
xlims = [-10, 10]
frac_train = 0.5

a_true = 1e0
group_dist_mat = np.array(
    [
        [0.0, 0.1, 1.0],
        [0.1, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)


def generate_separated_gp_data(n_groups, n_per_group, p=1):
    ## Generate data

    X_list = []
    Y_list = []
    groups_list = []
    # kernel = RBF()

    kernel_params = [np.log(1.0)] * 2
    kernel = lambda X1, X2: gp_kernel(kernel_params, X1, X2)

    for gg in range(n_groups):
        curr_X = np.random.uniform(low=xlims[0], high=xlims[1], size=(n_per_group, p))
        curr_K_XX = kernel(curr_X, curr_X) + noise_variance_true * np.eye(n_per_group)
        curr_Y = mvn.rvs(np.zeros(n_per_group), curr_K_XX)
        curr_groups = np.repeat(gg, n_per_group)

        X_list.append(curr_X)
        Y_list.append(curr_Y)
        groups_list.append(curr_groups)
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list)
    X_groups = np.concatenate(groups_list, axis=0).astype(int)

    return X, Y, X_groups


def generate_union_gp_data(n_groups, n_per_group, p=1):
    ## Generate data
    # kernel = RBF()
    kernel_params = [np.log(1.0)] * 2
    kernel = lambda X1, X2: gp_kernel(kernel_params, X1, X2)
    X = np.random.uniform(low=xlims[0], high=xlims[1], size=(n_groups * n_per_group, p))
    curr_K_XX = kernel(X, X) + noise_variance_true * np.eye(n_groups * n_per_group)
    Y = mvn.rvs(np.zeros(n_groups * n_per_group), curr_K_XX)
    # + np.random.normal(
    #     scale=np.sqrt(noise_variance_true), size=n_groups * n_per_group
    # )
    groups_list = []
    for gg in range(n_groups):
        # curr_groups_one_hot = np.zeros(n_groups)
        # curr_groups_one_hot[gg] = 1
        # curr_groups = np.repeat([curr_groups_one_hot], n_per_group, axis=0)
        curr_groups = np.repeat(gg, n_per_group)

        groups_list.append(curr_groups)
    X_groups = np.concatenate(groups_list, axis=0).astype(int)

    return X, Y, X_groups


def generate_hgp_data(n_groups, n_per_group, p=1):
    kernel = lambda params, X1, X2, groups1, groups2: hgp_kernel(
        kernel_params=params,
        x1=X1,
        x2=X2,
        groups1=groups1,
        groups2=groups2,
        within_group_kernel=gp_kernel,
        between_group_kernel=gp_kernel,
    )

    ## Generate data
    groups_list = []
    for gg in range(n_groups):
        # curr_groups_one_hot = np.zeros(n_groups)
        # curr_groups_one_hot[gg] = 1
        # curr_groups = np.repeat([curr_groups_one_hot], n_per_group, axis=0)
        curr_groups = np.repeat(gg, n_per_group)

        groups_list.append(curr_groups)
    X_groups = np.concatenate(groups_list, axis=0).astype(int)

    X = np.random.uniform(low=xlims[0], high=xlims[1], size=(n_groups * n_per_group, p))
    kernel_params = [np.log(1.0)] * 4
    # import ipdb; ipdb.set_trace()
    curr_K_XX = kernel(
        kernel_params, X, X, X_groups, X_groups
    ) + noise_variance_true * np.eye(n_groups * n_per_group)
    Y = mvn.rvs(np.zeros(n_groups * n_per_group), curr_K_XX)
    # + np.random.normal(
    #     scale=np.sqrt(noise_variance_true), size=n_groups * n_per_group
    # )

    return X, Y, X_groups


def generate_mggp_data(n_groups, n_per_group, p=1):

    kernel_params = [np.log(1.0)] * n_mgg_kernel_params
    if KERNEL_TYPE == "rbf":
        kernel_params[1] = np.log(a_true)
        kernel = lambda X1, X2, groups1, groups2: multigroup_rbf_kernel(
            kernel_params=kernel_params,
            x1=X1,
            x2=X2,
            groups1=groups1,
            groups2=groups2,
            group_distances=group_dist_mat,
        )

    elif KERNEL_TYPE == "matern":
        kernel_params[1] = np.log(a_true)
        kernel = lambda X1, X2, groups1, groups2: multigroup_matern12_covariance(
            kernel_params, X1, X2, groups1, groups2, group_dist_mat
        )

    ## Generate data
    groups_list = []
    for gg in range(n_groups):
        # curr_groups_one_hot = np.zeros(n_groups)
        # curr_groups_one_hot[gg] = 1
        # curr_groups = np.repeat([curr_groups_one_hot], n_per_group, axis=0)
        curr_groups = np.repeat(gg, n_per_group)

        groups_list.append(curr_groups)
    X_groups = np.concatenate(groups_list, axis=0).astype(int)

    X = np.random.uniform(low=xlims[0], high=xlims[1], size=(n_groups * n_per_group, p))

    curr_K_XX = kernel(X, X, X_groups, X_groups) + noise_variance_true * np.eye(
        n_groups * n_per_group
    )
    Y = mvn.rvs(np.zeros(n_groups * n_per_group), curr_K_XX)
    # + np.random.normal(
    #     scale=np.sqrt(noise_variance_true), size=n_groups * n_per_group
    # )

    return X, Y, X_groups


def separated_gp():

    errors_mggp = np.empty(n_repeats)
    errors_separated_gp = np.empty(n_repeats)
    errors_union_gp = np.empty(n_repeats)
    errors_hgp = np.empty(n_repeats)

    for ii in range(n_repeats):

        X, Y, X_groups = generate_separated_gp_data(
            n_groups=n_groups, n_per_group=n_per_group
        )

        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_groups_train,
            X_groups_test,
        ) = train_test_split(X, Y, X_groups, test_size=1 - frac_train, random_state=42)

        ############################
        ######### Fit MGGP #########
        ############################

        mggp = GP(kernel=mggp_kernel, is_mggp=True)
        group_dists = np.ones((n_groups, n_groups))
        np.fill_diagonal(group_dists, 0)
        mggp.fit(
            X_train,
            Y_train,
            groups=X_groups_train,
            group_distances=group_dists,
        )
        preds_mean = mggp.predict(X_test=X_test, groups_test=X_groups_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_mggp[ii] = curr_error

        ############################
        ##### Fit separated GP #####
        ############################

        sum_error_sep_gp = 0
        for groupnum in range(n_groups):
            curr_X_train = X_train[X_groups_train == groupnum]
            curr_Y_train = Y_train[X_groups_train == groupnum]
            curr_X_test = X_test[X_groups_test == groupnum]
            curr_Y_test = Y_test[X_groups_test == groupnum]

            sep_gp = GP(kernel=gp_kernel, is_mggp=False)

            sep_gp.fit(curr_X_train, curr_Y_train)
            preds_mean = sep_gp.predict(curr_X_test)
            curr_error_sep_gp = np.sum((curr_Y_test - preds_mean) ** 2)
            sum_error_sep_gp += curr_error_sep_gp

        errors_separated_gp[ii] = sum_error_sep_gp / Y_test.shape[0]

        ############################
        ####### Fit union GP #######
        ############################

        union_gp = GP(kernel=gp_kernel, is_mggp=False)
        union_gp.fit(X_train, Y_train)
        preds_mean = union_gp.predict(X_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_union_gp[ii] = curr_error

        ############################
        ######### Fit HGP ##########
        ############################
        hgp = GP(kernel=gp_kernel, within_group_kernel=gp_kernel, is_hgp=True)
        hgp.fit(X=X_train, y=Y_train, groups=X_groups_train)
        preds_mean = hgp.predict(X_test, X_groups_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_hgp[ii] = curr_error

    results_df = pd.melt(
        pd.DataFrame(
            {
                "MGGP": errors_mggp,
                "Separated GP": errors_separated_gp,
                "Union GP": errors_union_gp,
                "HGP": errors_hgp,
            }
        )
    )

    return results_df


def union_gp():

    errors_mggp = np.empty(n_repeats)
    errors_separated_gp = np.empty(n_repeats)
    errors_union_gp = np.empty(n_repeats)
    errors_hgp = np.empty(n_repeats)

    for ii in range(n_repeats):

        X, Y, X_groups = generate_union_gp_data(
            n_groups=n_groups, n_per_group=n_per_group
        )
        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_groups_train,
            X_groups_test,
        ) = train_test_split(X, Y, X_groups, test_size=1 - frac_train, random_state=42)

        ############################
        ######### Fit MGGP #########
        ############################

        mggp = GP(kernel=mggp_kernel, is_mggp=True)
        curr_dists = np.ones((n_groups, n_groups))
        np.fill_diagonal(curr_dists, 0)
        mggp.fit(
            X_train,
            Y_train,
            X_groups_train,
            group_distances=curr_dists,
        )
        preds_mean = mggp.predict(X_test, X_groups_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_mggp[ii] = curr_error

        ############################
        ##### Fit separated GP #####
        ############################

        sum_error_sep_gp = 0
        for groupnum in range(n_groups):
            curr_X_train = X_train[X_groups_train == groupnum]
            curr_Y_train = Y_train[X_groups_train == groupnum]
            curr_X_test = X_test[X_groups_test == groupnum]
            curr_Y_test = Y_test[X_groups_test == groupnum]

            sep_gp = GP(kernel=gp_kernel)

            sep_gp.fit(curr_X_train, curr_Y_train)
            preds_mean = sep_gp.predict(curr_X_test)
            curr_error_sep_gp = np.sum((curr_Y_test - preds_mean) ** 2)
            sum_error_sep_gp += curr_error_sep_gp

        errors_separated_gp[ii] = sum_error_sep_gp / Y_test.shape[0]

        ############################
        ####### Fit union GP #######
        ############################

        union_gp = GP(kernel=gp_kernel)
        union_gp.fit(X_train, Y_train)
        preds_mean = union_gp.predict(X_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_union_gp[ii] = curr_error

        ############################
        ######### Fit HGP ##########
        ############################
        hgp = GP(within_group_kernel=gp_kernel, kernel=gp_kernel, is_hgp=True)
        hgp.fit(X_train, Y_train, X_groups_train)
        preds_mean = hgp.predict(X_test, X_groups_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_hgp[ii] = curr_error

    results_df = pd.melt(
        pd.DataFrame(
            {
                "MGGP": errors_mggp,
                "Separated GP": errors_separated_gp,
                "Union GP": errors_union_gp,
                "HGP": errors_hgp,
            }
        )
    )

    return results_df


def hgp_experiment():

    errors_mggp = np.empty(n_repeats)
    errors_separated_gp = np.empty(n_repeats)
    errors_union_gp = np.empty(n_repeats)
    errors_hgp = np.empty(n_repeats)
    length_scale_shared = 1.0
    length_scale_group0 = 1.0
    length_scale_group1 = 1.0

    for ii in range(n_repeats):

        X, Y, X_groups = generate_hgp_data(n_groups=n_groups, n_per_group=n_per_group)
        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_groups_train,
            X_groups_test,
        ) = train_test_split(X, Y, X_groups, test_size=1 - frac_train, random_state=42)

        ############################
        ######### Fit MGGP #########
        ############################
        # hier_mg_kernel = lambda params, X1, X2, groups1, groups2, group_distances: hierarchical_multigroup_kernel(
        #     params, X1, X2, groups1, groups2, group_distances, within_group_kernel=rbf_covariance, between_group_kernel=rbf_covariance
        # )

        mggp = GP(kernel=mggp_kernel, is_mggp=True)
        curr_dists = np.ones((n_groups, n_groups))
        np.fill_diagonal(curr_dists, 0)
        mggp.fit(
            X_train,
            Y_train,
            X_groups_train,
            group_distances=curr_dists,
        )
        preds_mean = mggp.predict(X_test, X_groups_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_mggp[ii] = curr_error

        ############################
        ##### Fit separated GP #####
        ############################

        sum_error_sep_gp = 0
        for groupnum in range(n_groups):
            curr_X_train = X_train[X_groups_train == groupnum]
            curr_Y_train = Y_train[X_groups_train == groupnum]
            curr_X_test = X_test[X_groups_test == groupnum]
            curr_Y_test = Y_test[X_groups_test == groupnum]

            sep_gp = GP(kernel=gp_kernel)

            sep_gp.fit(curr_X_train, curr_Y_train)
            preds_mean = sep_gp.predict(curr_X_test)
            curr_error_sep_gp = np.sum((curr_Y_test - preds_mean) ** 2)
            sum_error_sep_gp += curr_error_sep_gp

        errors_separated_gp[ii] = sum_error_sep_gp / Y_test.shape[0]

        ############################
        ####### Fit union GP #######
        ############################

        union_gp = GP(kernel=gp_kernel)
        union_gp.fit(X_train, Y_train)
        preds_mean = union_gp.predict(X_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_union_gp[ii] = curr_error

        ############################
        ######### Fit HGP ##########
        ############################
        hgp = GP(within_group_kernel=gp_kernel, kernel=gp_kernel, is_hgp=True)
        hgp.fit(X_train, Y_train, X_groups_train)
        preds_mean = hgp.predict(X_test, X_groups_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_hgp[ii] = curr_error

    results_df = pd.melt(
        pd.DataFrame(
            {
                "MGGP": errors_mggp,
                "Separated GP": errors_separated_gp,
                "Union GP": errors_union_gp,
                "HGP": errors_hgp,
            }
        )
    )

    return results_df


def mggp():

    errors_mggp = np.empty(n_repeats)
    errors_separated_gp = np.empty(n_repeats)
    errors_union_gp = np.empty(n_repeats)
    errors_hgp = np.empty(n_repeats)

    for ii in range(n_repeats):

        X, Y, X_groups = generate_mggp_data(n_groups=n_groups, n_per_group=n_per_group)
        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_groups_train,
            X_groups_test,
        ) = train_test_split(X, Y, X_groups, test_size=1 - frac_train, random_state=42)

        ############################
        ######### Fit MGGP #########
        ############################

        mggp = GP(kernel=mggp_kernel, is_mggp=True)
        mggp.fit(X_train, Y_train, X_groups_train, group_dist_mat)
        preds_mean = mggp.predict(X_test, X_groups_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_mggp[ii] = curr_error

        ############################
        ##### Fit separated GP #####
        ############################

        sum_error_sep_gp = 0
        for groupnum in range(n_groups):
            curr_X_train = X_train[X_groups_train == groupnum]
            curr_Y_train = Y_train[X_groups_train == groupnum]
            curr_X_test = X_test[X_groups_test == groupnum]
            curr_Y_test = Y_test[X_groups_test == groupnum]

            sep_gp = GP(kernel=gp_kernel)

            sep_gp.fit(curr_X_train, curr_Y_train)
            preds_mean = sep_gp.predict(curr_X_test)
            curr_error_sep_gp = np.sum((curr_Y_test - preds_mean) ** 2)
            sum_error_sep_gp += curr_error_sep_gp

        errors_separated_gp[ii] = sum_error_sep_gp / Y_test.shape[0]

        ############################
        ####### Fit union GP #######
        ############################

        union_gp = GP(kernel=gp_kernel)
        union_gp.fit(X_train, Y_train)
        preds_mean = union_gp.predict(X_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_union_gp[ii] = curr_error

        ############################
        ######### Fit HGP ##########
        ############################
        hgp = GP(within_group_kernel=gp_kernel, kernel=gp_kernel, is_hgp=True)
        hgp.fit(X_train, Y_train, X_groups_train)
        preds_mean = hgp.predict(X_test, X_groups_test)
        curr_error = np.mean((Y_test - preds_mean) ** 2)
        errors_hgp[ii] = curr_error

    results_df = pd.melt(
        pd.DataFrame(
            {
                "MGGP": errors_mggp,
                "Separated GP": errors_separated_gp,
                "Union GP": errors_union_gp,
                "HGP": errors_hgp,
            }
        )
    )

    return results_df


if __name__ == "__main__":

    sep_gp_results = separated_gp()
    union_gp_results = union_gp()
    hgp_results = hgp_experiment()
    mggp_results = mggp()
    # import ipdb; ipdb.set_trace()

    sep_gp_results["method"] = ["Separated GP"] * sep_gp_results.shape[0]
    union_gp_results["method"] = ["Union GP"] * union_gp_results.shape[0]
    hgp_results["method"] = ["HGP"] * hgp_results.shape[0]
    mggp_results["method"] = [
        r"MGGP, $a=" + str(round(a_true, 1)) + "$"
    ] * mggp_results.shape[0]

    all_results = pd.concat(
        [sep_gp_results, union_gp_results, hgp_results, mggp_results], axis=0
    )
    cols = all_results.columns.values
    cols[cols == "variable"] = "Fitted model"
    all_results.columns = cols

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=all_results, x="method", y="value", hue="Fitted model")
    plt.xlabel("Model from which data is generated")
    plt.ylabel("Prediction MSE")
    plt.yscale("log")
    plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()

    if KERNEL_TYPE == "rbf":
        plt.savefig("../../plots/prediction_simulation_experiment_rbf.png")
    elif KERNEL_TYPE == "matern":
        plt.savefig("../../plots/prediction_simulation_experiment_matern.png")
    else:
        plt.savefig("../../plots/prediction_simulation_experiment.png")
    plt.show()

    import ipdb

    ipdb.set_trace()
