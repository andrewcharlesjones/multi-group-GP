import matplotlib
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pcpca import PCPCA
import numpy as np
from os.path import join as pjoin
import pickle
from sklearn.model_selection import train_test_split
from multigroupGP import MultiGroupRBF
import arviz as az
import pandas as pd
import seaborn as sns
import pickle

import socket

import stan

if socket.gethostname() == "andyjones":
    MODEL_DIR = "../../multigroupGP/models"
else:
    MODEL_DIR = "../../models"


font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

N_MCMC_ITER = 30
N_WARMUP_ITER = 30
N_CHAINS = 4

frac_train = 0.75
n0, n1 = 10, 10
ntotal = n0 + n1
noise_variance = 0.1
p = 2
n_groups = 2
# n_infinite = 80

grid_size = 10
limits = [-5, 5]
x1s = np.linspace(*limits, num=grid_size)
x2s = np.linspace(*limits, num=grid_size)
X1, X2 = np.meshgrid(x1s, x2s)
X_full_one_group = np.vstack([X1.ravel(), X2.ravel()]).T
n_infinite_one_group = X_full_one_group.shape[0]
X_full = np.concatenate([X_full_one_group, X_full_one_group], axis=0)
n_infinite = X_full.shape[0]

X0_group_one_hot = np.zeros(n_groups)
X0_group_one_hot[0] = 1
X0_groups = np.repeat([X0_group_one_hot], n_infinite // 2, axis=0)
X1_group_one_hot = np.zeros(n_groups)
X1_group_one_hot[1] = 1
X1_groups = np.repeat([X1_group_one_hot], n_infinite // 2, axis=0)
groups_oh_full = np.concatenate([X0_groups, X1_groups], axis=0)
groups_full = np.concatenate(
    [np.zeros(n_infinite // 2), np.ones(n_infinite // 2)]
).astype(int)

# X_full = np.random.uniform(low=-5, high=5, size=(n_infinite, p))

# X_full = X_full[
#     np.random.choice(np.arange(n_infinite), size=n_infinite, replace=True), :
# ]

sigma2_true = 1.0
a_true = 1.0
b_true = 1.0
mean_intercepts = np.array([1, 2])

groups_dists = np.ones((2, 2))
np.fill_diagonal(groups_dists, 0)

kernel = MultiGroupRBF(
    amplitude=sigma2_true, group_diff_param=a_true, lengthscale=b_true
)
K_XX = kernel(
    x1=X_full,
    x2=X_full,
    groups1=groups_full,
    groups2=groups_full,
    group_distances=groups_dists,
)

Y_full_noiseless = mvn(np.zeros(n_infinite), K_XX + np.eye(n_infinite) * 1e-5).rvs()
Y_means = groups_oh_full @ mean_intercepts
Y_full_noiseless_with_mean = Y_full_noiseless + Y_means
Y_full = Y_full_noiseless_with_mean + np.random.normal(
    scale=np.sqrt(noise_variance), size=n_infinite
)


group0_idx = np.random.choice(np.arange(n_infinite // 2), size=n0)
group1_idx = np.random.choice(np.arange(n_infinite // 2, n_infinite), size=n1)
X = np.concatenate([X_full[group0_idx], X_full[group1_idx]])
Y = np.concatenate([Y_full[group0_idx], Y_full[group1_idx]])
groups = np.concatenate([groups_full[group0_idx], groups_full[group1_idx]])
groups_oh = np.concatenate([groups_oh_full[group0_idx], groups_oh_full[group1_idx]])

(
    Xtrain,
    Xtest,
    Ytrain,
    Ytest,
    groups_train,
    groups_test,
    groups_oh_train,
    groups_oh_test,
) = train_test_split(X, Y, groups, groups_oh, test_size=1 - frac_train, random_state=42)
Xtest = X_full
Ytest = Y_full
groups_test = groups_full
ntrain, ntest = Xtrain.shape[0], Xtest.shape[0]

data = {
    "P": p,
    "ngroups": n_groups,
    "N1": ntrain,
    "N2": ntest,
    "x1": Xtrain.squeeze(),
    "x2": Xtest.squeeze(),
    "design1": groups_oh_train,
    "design2": groups_oh_full,
    "y1": Ytrain,
    "groups1": groups_train,
    "groups2": groups_test,
    "group_dist_mat": 1 - np.eye(n_groups),
}

#

## Load model
with open(pjoin(MODEL_DIR, "mggp_multivariable.stan"), "r") as file:
    model_code = file.read()

# Set up model
posterior = stan.build(model_code, data=data)


# Start sampling
fit = posterior.sample(
    num_chains=N_CHAINS, num_warmup=N_WARMUP_ITER, num_samples=N_MCMC_ITER
)

# with open('./out/stan_out/mggp_stan_fit.pickle', 'wb') as handle:
#     pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

pd.DataFrame(Xtrain).to_csv("./out/stan_out/X_train.csv")
pd.DataFrame(Ytrain).to_csv("./out/stan_out/Y_train.csv")
pd.DataFrame(groups_train).to_csv("./out/stan_out/groups_train.csv")

pd.DataFrame(X_full).to_csv("./out/stan_out/X_full.csv")
pd.DataFrame(Y_full).to_csv("./out/stan_out/Y_full.csv")
pd.DataFrame(groups_full).to_csv("./out/stan_out/groups_full.csv")

percentiles = [2.5, 50, 97.5]
percentiles_df = pd.DataFrame(
    {
        "alpha": np.percentile(fit["alpha"].squeeze(), q=percentiles),
        "lengthscale": np.percentile(fit["lengthscale"].squeeze(), q=percentiles),
        "outputvariance": np.percentile(fit["outputvariance"].squeeze(), q=percentiles),
        "sigma": np.percentile(fit["sigma"].squeeze(), q=percentiles),
        "beta1": np.percentile(fit["beta"][0], q=percentiles),
        "beta2": np.percentile(fit["beta"][1], q=percentiles),
    }
)


truth_df = pd.DataFrame(
    {
        "alpha": a_true,
        "lengthscale": b_true,
        "outputvariance": sigma2_true,
        "sigma": noise_variance,
        "beta1": mean_intercepts[0],
        "beta2": mean_intercepts[1],
    },
    index=["Truth"],
)
percentiles_df = pd.concat([truth_df, percentiles_df], axis=0)
percentiles_df.index = ["Truth", "p2.5", "p50", "p97.5"]

percentiles_df.to_csv("./out/stan_out/parameter_posterior_percentiles.csv")


percentiles_latent_process = pd.DataFrame(
    # np.percentile(fit["f"][ntrain:, :], percentiles, axis=1),
    np.vstack(
        [
            np.mean(fit["f"][ntrain:, :], axis=1),
            np.std(fit["f"][ntrain:, :], axis=1),
        ]
    )
)

percentiles_predictive_process = pd.DataFrame(
    # np.percentile(fit["y2"], percentiles, axis=1),
    np.vstack(
        [
            np.mean(fit["y2"], axis=1),
            np.std(fit["y2"], axis=1),
        ]
    )
)

assert percentiles_latent_process.shape[1] == percentiles_predictive_process.shape[1]

percentiles_latent_process.to_csv("./out/stan_out/latent_process_percentiles.csv")
percentiles_predictive_process.to_csv(
    "./out/stan_out/predictive_process_percentiles.csv"
)


import ipdb

ipdb.set_trace()

arviz_summary = az.summary(fit)


# import ipdb; ipdb.set_trace()
plt.figure(figsize=(12, 5))

plt.scatter(
    Xtrain[groups_train == 0, 0],
    Ytrain[groups_train == 0],
    color="blue",  # , label="Group 1"
)
plt.scatter(
    Xtrain[groups_train == 1, 0],
    Ytrain[groups_train == 1],
    color="red",  # , label="Group 2"
)

Ytest_samples = fit["y2"]
Ftest_samples = fit["f"][ntrain:, :]

## Group 1
colors = ["blue", "red"]
for groupnum in [0, 1]:
    curr_idx = np.where(groups_test == groupnum)[0]

    ## Plot F
    curr_Ftest_samples = Ftest_samples[curr_idx, :]
    curr_Xtest = Xtest[curr_idx, 0]
    preds_mean = curr_Ftest_samples.mean(1)
    preds_stddev = curr_Ftest_samples.std(1)
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

    ## Plot Y
    curr_Ytest_samples = Ytest_samples[curr_idx, :]
    curr_Xtest = Xtest[curr_idx, 0]
    preds_mean = curr_Ytest_samples.mean(1)
    preds_stddev = curr_Ytest_samples.std(1)
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


plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.tight_layout()
# plt.savefig("../../plots/mggp_predictive_samples.png")
plt.show()
plt.close()
import ipdb

ipdb.set_trace()

samples_df = pd.DataFrame(
    {
        r"$a$": fit["alpha"].squeeze(),
        r"$b$": fit["lengthscale"].squeeze(),
        r"$\sigma^2$": fit["outputvariance"].squeeze(),
        r"$\beta_1$": fit["beta"][0, :].squeeze(),
        r"$\beta_2$": fit["beta"][1, :].squeeze(),
        r"$\tau^2$": fit["sigma"].squeeze() ** 2,
    }
)
samples_df_melted = pd.melt(samples_df)
truth_df = pd.DataFrame(
    {
        r"$a$": [a_true],
        r"$b$": [b_true],
        r"$\sigma^2$": [sigma2_true],
        r"$\beta_1$": [mean_intercepts[0]],
        r"$\beta_2$": [mean_intercepts[1]],
        r"$\tau^2$": noise_variance,
    }
)
truth_df_melted = pd.melt(truth_df)
plt.figure(figsize=(7, 5))
sns.boxplot(data=samples_df_melted, y="variable", x="value", orient="h", color="gray")
sns.pointplot(
    data=truth_df_melted,
    y="variable",
    x="value",
    join=False,
    orient="h",
    marker="x",
    color="red",
)
plt.ylabel("")
plt.xlabel("Samples")
plt.tight_layout()
plt.savefig("../../plots/mggp_hyperparameter_samples.png")
plt.close()

az.plot_autocorr(fit, var_names=["lengthscale", "outputvariance", "alpha"])
plt.tight_layout()
plt.savefig("../../plots/mggp_autocorrelation.png")
# plt.show()
plt.close()

import ipdb

ipdb.set_trace()
