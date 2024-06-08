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
from sklearn.model_selection import train_test_split
from multigroupGP import MultiGroupRBF
import arviz as az
import pandas as pd
import seaborn as sns
from sklearn.gaussian_process.kernels import RBF

import socket

if socket.gethostname() == "andyjones":
    MODEL_DIR = "../../multigroupGP/models"
else:
    MODEL_DIR = "../../models"


font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

N_MCMC_ITER = 100
N_WARMUP_ITER = 100
N_CHAINS = 4

n = 40
noise_variance = 0.1
p = 1

limits = [-5, 5]

# X = np.random.uniform(*limits, size=(n, p))
X = np.linspace(*limits, n).reshape(-1, 1)

sigma2_true = 1.0
b_true = 1.0
mean_intercept = 0.0


kernel = RBF()

K_XX = kernel(X)

Y_noiseless = mvn(np.zeros(n), K_XX + np.eye(n) * 1e-8).rvs()
Y_noiseless_with_mean = Y_noiseless + mean_intercept
Y = Y_noiseless_with_mean + np.random.normal(scale=np.sqrt(noise_variance), size=n)

data = {
    "N": n,
    "x": X.squeeze(),
    "y": Y,
}


## Load model
with open(pjoin(MODEL_DIR, "gp-fit.stan"), "r") as file:
    model_code = file.read()

# Set up model
posterior = stan.build(model_code, data=data)

# Start sampling
fit = posterior.sample(
    num_chains=N_CHAINS, num_warmup=N_WARMUP_ITER, num_samples=N_MCMC_ITER
)

# plt.scatter(X, Y)
# plt.scatter(X, mvn.rvs(fit["mu"].mean() * np.ones(n), fit["K"][:, :, -1]))
# plt.show()

# import ipdb; ipdb.set_trace()
n_mcmc_samples = fit["L_K"].shape[-1]

## Sample y
y_samples = np.array(
    [fit["L_K"][:, :, ii] @ np.random.normal(size=n) for ii in range(n_mcmc_samples)]
)

## Sample latent process
f_samples = np.zeros((n_mcmc_samples, n))
y_samples = np.zeros((n_mcmc_samples, n))
for ii in range(n_mcmc_samples):
    curr_K_noisy = fit["K_noisy"][:, :, ii]
    curr_K = fit["K"][:, :, ii]
    K_noisy_inv = np.linalg.solve(curr_K_noisy, np.eye(n))
    curr_sigma = fit["sigma"].squeeze()[ii] ** 2
    # K_inv_plus_D_inv = K_inv + 1 / curr_sigma * np.eye(n)
    # M = np.linalg.solve(K_inv_plus_D_inv, np.eye(n))
    curr_m = Y - fit["mu"][:, ii] * np.ones(n)
    curr_premult_K = curr_K @ K_noisy_inv

    f_samples[ii] = mvn.rvs(mean=curr_premult_K @ curr_m, cov=curr_premult_K @ curr_K)
    y_samples[ii] = f_samples[ii] + fit["mu"][:, ii] * np.ones(n)


plt.scatter(X, Y, alpha=0.3)

f_samples_mean = f_samples.mean(0)
y_samples_mean = y_samples.mean(0)
means_mean = fit["mu"].mean(1)

# plt.scatter(X[:n0], f_samples_mean[:n0] + means_mean[:n0], color="blue")
# plt.scatter(X[n0:], f_samples_mean[n0:] + means_mean[n0:], color="red")
plt.scatter(X, y_samples_mean)
plt.scatter(X, f_samples_mean)

plt.show()

import ipdb

ipdb.set_trace()

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

plt.figure(figsize=(12, 5))

plt.scatter(
    Xtrain[groups_train == 0],
    Ytrain[groups_train == 0],
    color="blue",  # , label="Group 1"
)
plt.scatter(
    Xtrain[groups_train == 1],
    Ytrain[groups_train == 1],
    color="red",  # , label="Group 2"
)

Ytest_samples = fit["y2"]
Ftest_samples = fit["f"][ntrain:, :]
# import ipdb; ipdb.set_trace()

## Group 1
colors = ["blue", "red"]
for groupnum in [0, 1]:
    curr_idx = np.where(groups_test == groupnum)[0]

    ## Plot F
    curr_Ftest_samples = Ftest_samples[curr_idx, :]
    curr_Xtest = Xtest[curr_idx, :]
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
    curr_Xtest = Xtest[curr_idx, :]
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
plt.savefig("../../plots/mggp_predictive_samples.png")
# plt.show()
plt.close()

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
