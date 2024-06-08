import matplotlib
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pcpca import PCPCA
import numpy as np
import stan
from hashlib import md5
from os.path import join as pjoin
import pickle
import os
from multigroupGP import MultiGroupRBF
import arviz as az
import pandas as pd
import seaborn as sns

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


X = pd.read_csv("./out/stan_out/X_train.csv", index_col=0).values
Y = pd.read_csv("./out/stan_out/Y_train.csv", index_col=0).values
groups = pd.read_csv("./out/stan_out/groups_train.csv", index_col=0).values.squeeze()
n_groups = len(np.unique(groups))

cov_params_df = pd.read_csv("./out/stan_out/cov_params_samples.csv", index_col=0)
noise_variance_df = pd.read_csv(
    "./out/stan_out/noise_variance_samples.csv", index_col=0
)
beta_df = pd.read_csv("./out/stan_out/beta_samples.csv", index_col=0)

n = X.shape[0]


def multigroup_RBF(
    x, groups, sample_group_dist_mat, lengthscale, outputvariance, alpha
):
    squared_dist_mat = (x - x.T) ** 2
    scaling_term = 1 / (sample_group_dist_mat * alpha ** 2 + 1) ** 0.5
    K = scaling_term * np.exp(
        -0.5 * squared_dist_mat * lengthscale / (sample_group_dist_mat * alpha ** 2 + 1)
    )
    K += 1e-8 * np.eye(n)
    return K


group_dist_mat = 1 - np.eye(n_groups)
sample_group_dist_mat = np.zeros((n, n))
for ii in range(n):
    for jj in range(ii, n):
        sample_group_dist_mat[ii, jj] = group_dist_mat[groups[ii], groups[jj]]
        sample_group_dist_mat[jj, ii] = sample_group_dist_mat[ii, jj]


n_mcmc_samples = len(cov_params_df)


percentiles = [2.5, 50, 97.5]

## Save table of posterior credible intervals
percentiles_df = pd.DataFrame(
    {
        "$\\alpha$": np.percentile(cov_params_df["alpha"].values, q=percentiles),
        "b": np.percentile(cov_params_df["lengthscale"].values, q=percentiles),
        "$\\sigma^2$": np.percentile(
            cov_params_df["outputvariance"].values, q=percentiles
        ),
        "$\\tau_1$": np.percentile(noise_variance_df.iloc[:, 0].values, q=percentiles),
        "$\\tau_2$": np.percentile(noise_variance_df.iloc[:, 1].values, q=percentiles),
        "$\\beta_1$": np.percentile(beta_df.iloc[0, :], q=percentiles),
        "$\\beta_2$": np.percentile(beta_df.iloc[1, :], q=percentiles),
    }
)

true_params = np.array(
    [
        1.0,
        1.0,
        1.0,
        0.1,
        0.3,
        1.0,
        2.0,
    ]
)
true_params_df = pd.DataFrame(
    true_params.reshape(1, -1), columns=percentiles_df.columns
)
percentiles_df = pd.concat([percentiles_df, true_params_df])


table_df = pd.DataFrame(np.zeros((percentiles_df.shape[1], 3)))
table_df.iloc[:, 0] = percentiles_df.columns.values
table_df.iloc[:, 1] = percentiles_df.iloc[3, :].values
for ii in range(percentiles_df.shape[1]):
    table_df.iloc[ii, 2] = (
        str(round(percentiles_df.iloc[1, ii], 2))
        + " ("
        + str(round(percentiles_df.iloc[0, ii], 2))
        + "; "
        + str(round(percentiles_df.iloc[2, ii], 2))
        + ")"
    )
# import ipdb; ipdb.set_trace()

# table_df.index = percentiles_df.columns
# table_df.columns = ["Parameter", "Truth", "Posteriorpercentiles"]

# table_df.to_csv("./out/stan_out/posterior_percentiles_table.csv", index=False, sep=",")

table_df.columns = [
    "\\textbf{Parameter}",
    "\\textbf{True}",
    "\\textbf{Posterior percentiles}",
]

with open("./out/stan_out/posterior_percentiles_table.txt", "w") as ff:

    ff.write(
        table_df.columns.values[0]
        + " & "
        + table_df.columns.values[1]
        + " & "
        + table_df.columns.values[2]
        + " \\\ \n"
    )
    ff.write("\\hline\n")

    for ii in range(len(table_df)):

        ff.write(
            str(table_df.values[ii, 0])
            + " & "
            + str(table_df.values[ii, 1])
            + " & "
            + str(table_df.values[ii, 2])
            + " \\\ \n"
        )
        ff.write("\\hline\n")


# import ipdb; ipdb.set_trace()


## Sample latent process
f_samples = np.zeros((n_mcmc_samples, n))
y_samples = np.zeros((n_mcmc_samples, n))
for ii in range(n_mcmc_samples):

    curr_outputvariance, curr_lengthscale, curr_alpha = cov_params_df.iloc[ii].values

    curr_sigma = noise_variance_df.values.squeeze()[ii]
    curr_beta = beta_df.values[:, ii]
    curr_K = multigroup_RBF(
        X,
        groups,
        sample_group_dist_mat,
        outputvariance=curr_outputvariance,
        lengthscale=curr_lengthscale,
        alpha=curr_alpha,
    )

    curr_K_noisy = curr_K + np.diag(curr_sigma[groups])
    K_noisy_inv = np.linalg.solve(curr_K_noisy, np.eye(n))

    curr_mean = curr_beta[groups]
    curr_m = Y.squeeze() - curr_mean
    curr_premult_K = curr_K @ K_noisy_inv
    # import ipdb; ipdb.set_trace()

    f_samples[ii] = mvn.rvs(mean=curr_premult_K @ curr_m, cov=curr_premult_K @ curr_K)
    y_samples[ii] = f_samples[ii] + curr_mean


f_samples_mean = f_samples.mean(0)
y_samples_mean = y_samples.mean(0)
# means_mean = fit["means"].mean(1)

# colors = ["blue", "red"]
# for ii in range(2):
#     curr_idx = np.where(groups == ii)
#     plt.scatter(X[curr_idx], Y[curr_idx], color=colors[ii], alpha=0.3)
#     plt.scatter(X[curr_idx], y_samples_mean[curr_idx], color=colors[ii])
# plt.show()


plt.figure(figsize=(12, 10))


plt.subplot(211)
plt.title(r"Latent process $Z$")
## Group 1
colors = ["blue", "red"]
for groupnum in [0, 1]:
    curr_idx = np.where(groups == groupnum)[0]

    ## Plot F
    curr_Ftest_samples = f_samples[:, curr_idx]
    curr_Xtest = X[curr_idx, :]
    preds_mean = curr_Ftest_samples.mean(0)
    preds_stddev = curr_Ftest_samples.std(0)
    preds_upper = preds_mean + 2 * preds_stddev
    preds_lower = preds_mean - 2 * preds_stddev
    sorted_idx = np.argsort(curr_Xtest.squeeze())
    plt.plot(
        curr_Xtest.squeeze()[sorted_idx],
        preds_mean[sorted_idx],
        c=colors[groupnum],
        alpha=0.5,
        label=r"$F(x; c_" + str(groupnum + 1) + ")$",
        linestyle="-",
    )
    plt.fill_between(
        curr_Xtest.squeeze()[sorted_idx],
        preds_lower[sorted_idx],
        preds_upper[sorted_idx],
        alpha=0.3,
        color=colors[groupnum],
    )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ylims1 = plt.gca().get_ylim()


plt.subplot(212)
plt.title("Response $y$")
plt.scatter(
    X[groups == 0],
    Y[groups == 0],
    color="blue",  # , label="Group 1"
)
plt.scatter(
    X[groups == 1],
    Y[groups == 1],
    color="red",  # , label="Group 2"
)

for groupnum in [0, 1]:

    curr_idx = np.where(groups == groupnum)[0]

    ## Plot Y
    curr_Ytest_samples = y_samples[:, curr_idx]
    curr_Xtest = X[curr_idx, :]
    preds_mean = curr_Ytest_samples.mean(0)
    preds_stddev = curr_Ytest_samples.std(0)
    preds_upper = preds_mean + 2 * preds_stddev
    preds_lower = preds_mean - 2 * preds_stddev
    sorted_idx = np.argsort(curr_Xtest.squeeze())
    plt.plot(
        curr_Xtest.squeeze()[sorted_idx],
        preds_mean[sorted_idx],
        c=colors[groupnum],
        alpha=0.5,
        label=r"$Y(x; c_" + str(groupnum + 1) + ")$",
        linestyle="--",
    )
    plt.fill_between(
        curr_Xtest.squeeze()[sorted_idx],
        preds_lower[sorted_idx],
        preds_upper[sorted_idx],
        alpha=0.3,
        color=colors[groupnum],
    )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

ylims2 = plt.gca().get_ylim()
ylim_lower = min(ylims1[0], ylims2[0])
ylim_upper = min(ylims1[1], ylims2[1])

plt.subplot(211)
plt.ylim([ylim_lower, ylim_upper])
plt.subplot(212)
plt.ylim([ylim_lower, ylim_upper])
plt.tight_layout()

plt.savefig("./out/stan_out/latent_and_predictive_processes.png", dpi=300)
plt.show()
import ipdb

ipdb.set_trace()

# X_full = pd.read_csv("./out/stan_out/X_full.csv", index_col=0).values
# groups_full = pd.read_csv(
#     "./out/stan_out/groups_full.csv", index_col=0
# ).values.squeeze()

# percentiles_df = pd.read_csv(
#     "./out/stan_out/parameter_posterior_percentiles.csv", index_col=0
# )
# percentiles_latent_process_df = pd.read_csv(
#     "./out/stan_out/latent_process_percentiles.csv", index_col=0
# )
# percentiles_latent_process_train_df = pd.read_csv(
#     "./out/stan_out/latent_process_percentiles_train.csv", index_col=0
# )
# percentiles_predictive_process = pd.read_csv(
#     "./out/stan_out/predictive_process_percentiles.csv", index_col=0
# )
# percentiles_predictive_process_train = pd.read_csv(
#     "./out/stan_out/predictive_process_percentiles_train.csv", index_col=0
# )

# ypred_means = percentiles_predictive_process.values[0]
# ypred_stddevs = percentiles_predictive_process.values[1]

# ytrain_means = percentiles_predictive_process_train.values[0]
# ytrain_stddevs = percentiles_predictive_process_train.values[1]

# ftrain_means = percentiles_latent_process_train_df.values[0]
# ftrain_stddevs = percentiles_latent_process_train_df.values[1]


# plt.figure(figsize=(30, 24))
# n_rows = 5
# n_cols = 3
# for gg_main in range(n_groups):

#     plt.subplot(n_rows, n_cols, gg_main + 1)

#     for gg in range(n_groups):
#         curr_idx = np.where(groups_train == gg)[0]
#         curr_idx_full = np.where(groups_full == gg)[0]

#         plt.scatter(Xtrain[curr_idx], Ytrain[curr_idx], color="gray")

#     curr_idx_main = np.where(groups_full == gg_main)[0]

#     curr_idx_train_main = np.where(groups_train == gg_main)[0]
#     curr_sorted_idx = np.argsort(Xtrain[curr_idx_train_main].squeeze())

#     plt.fill_between(
#         Xtrain[curr_idx_train_main][curr_sorted_idx].squeeze(),
#         ytrain_means[curr_idx_train_main][curr_sorted_idx]
#         - 2 * ytrain_stddevs[curr_idx_train_main][curr_sorted_idx],
#         ytrain_means[curr_idx_train_main][curr_sorted_idx]
#         + 2 * ytrain_stddevs[curr_idx_train_main][curr_sorted_idx],
#         alpha=0.2,
#         color="red",
#     )

#     plt.plot(
#         Xtrain[curr_idx_train_main][curr_sorted_idx],
#         ytrain_means[curr_idx_train_main][curr_sorted_idx],
#         color="red",
#         linestyle="--",
#         linewidth=4,
#     )

#     plt.scatter(Xtrain[curr_idx_train_main], Ytrain[curr_idx_train_main], color="red")

#     plt.title(tissue_labels[gg_main])

#     plt.xlabel("TXNIP expression")
#     plt.ylabel("Ischemic time")
#     # plt.legend()
#     # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#     plt.tight_layout()
# plt.savefig(
#     "./out/group_specific_posterior_plots/gtex_posterior_inference_all_groups_subplots.png".format(
#         tissue_labels[gg_main].replace(" ", "_")
#     )
# )
# plt.show()
# plt.close()


# plt.figure(figsize=(17, 6))
# for gg in range(n_groups):
#     curr_idx_train_main = np.where(groups_train == gg)[0]
#     curr_sorted_idx = np.argsort(Xtrain[curr_idx_train_main].squeeze())

#     plt.plot(
#         Xtrain[curr_idx_train_main][curr_sorted_idx],
#         ytrain_means[curr_idx_train_main][curr_sorted_idx],
#         # color="red",
#         # linestyle="--",
#         # linewidth=4,
#         label=tissue_labels[gg],
#     )

#     plt.scatter(Xtrain[curr_idx_train_main], Ytrain[curr_idx_train_main])

# plt.xlabel("TXNIP expression")
# plt.ylabel("Ischemic time")
# plt.legend()
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# plt.savefig(
#     "./out/group_specific_posterior_plots/gtex_posterior_inference_all_groups.png"
# )
# plt.show()
