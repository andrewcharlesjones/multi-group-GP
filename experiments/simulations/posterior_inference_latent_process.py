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

import socket

if socket.gethostname() == "andyjones":
    MODEL_DIR = "../../multigroupGP/models"
else:
    MODEL_DIR = "../../models"


font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

N_MCMC_ITER = 200
N_WARMUP_ITER = 100
N_CHAINS = 4

n0, n1 = 50, 50
n = n0 + n1
noise_variance = np.array([0.1, 0.3])
p = 1
n_groups = 2

limits = [-5, 5]

X = np.random.uniform(*limits, size=(n, p))

X0_group_one_hot = np.zeros(n_groups)
X0_group_one_hot[0] = 1
X0_groups = np.repeat([X0_group_one_hot], n0, axis=0)
X1_group_one_hot = np.zeros(n_groups)
X1_group_one_hot[1] = 1
X1_groups = np.repeat([X1_group_one_hot], n1, axis=0)
groups_oh = np.concatenate([X0_groups, X1_groups], axis=0)
groups = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)


sigma2_true = 1.0
a_true = 1.0
b_true = 1.0
mean_intercepts = np.array([1, 2])

groups_dists = 1 - np.eye(n_groups)

kernel = MultiGroupRBF(
    amplitude=sigma2_true, group_diff_param=a_true, lengthscale=b_true
)

K_XX = kernel(
    x1=X,
    x2=X,
    groups1=groups,
    groups2=groups,
    group_distances=groups_dists,
)

# Y_noiseless = mvn(np.zeros(n), K_XX + np.eye(n) * 1e-6).rvs()
# Y_means = groups_oh @ mean_intercepts
# Y_noiseless_with_mean = Y_noiseless + Y_means
# Y = Y_noiseless_with_mean + np.random.normal(
#     scale=np.sqrt(noise_variance), size=n
# )
Y_means = groups_oh @ mean_intercepts
Y = mvn(Y_means, K_XX + np.eye(n) * noise_variance[groups]).rvs()

data = {
    "N": n,
    "P": p,
    "x": X,
    "design": groups_oh,
    "y": Y,
    "groups": groups,
    "ngroups": n_groups,
    "group_dist_mat": groups_dists,
}


## Load model
with open(pjoin(MODEL_DIR, "mggp_collapsed.stan"), "r") as file:
    model_code = file.read()

# Set up model
posterior = stan.build(model_code, data=data)

# Start sampling
fit = posterior.sample(
    num_chains=N_CHAINS, num_warmup=N_WARMUP_ITER, num_samples=N_MCMC_ITER
)

arviz_summary = az.summary(fit)


cov_params_df = pd.DataFrame(
    {
        "outputvariance": fit["outputvariance"].squeeze(),
        "lengthscale": fit["lengthscale"].squeeze(),
        "alpha": fit["alpha"].squeeze(),
    }
)
noise_variance_df = pd.DataFrame(fit["sigma"].T)
beta_df = pd.DataFrame(fit["beta"])

cov_params_df.to_csv("./out/stan_out/cov_params_samples.csv")
noise_variance_df.to_csv("./out/stan_out/noise_variance_samples.csv")
beta_df.to_csv("./out/stan_out/beta_samples.csv")

# fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# [ax.set_xlabel("Lag") for ax in axs]
# axs[0].set_ylabel("Correlation")
# az.plot_autocorr(fit, var_names="alpha", ax=axs)
# [axs[ii].set_title(r"$a$, Chain " + str(ii + 1)) for ii in range(len(axs))]
# plt.tight_layout()
# plt.show()

print(cov_params_df.mean())

axs = az.plot_autocorr(fit, combined=True, textsize=20, figsize=(15, 15))
axs[0, 0].set_title(r"$\sigma^2$")
axs[0, 0].set_xlabel("Lag")
axs[0, 0].set_ylabel("Correlation")

axs[0, 1].set_title(r"$b$")
axs[0, 1].set_xlabel("Lag")
axs[0, 1].set_ylabel("Correlation")

axs[0, 2].set_title(r"$a$")
axs[0, 2].set_xlabel("Lag")
axs[0, 2].set_ylabel("Correlation")

axs[1, 0].set_title(r"$\tau^2_1$")
axs[1, 0].set_xlabel("Lag")
axs[1, 0].set_ylabel("Correlation")

axs[1, 1].set_title(r"$\tau^2_2$")
axs[1, 1].set_xlabel("Lag")
axs[1, 1].set_ylabel("Correlation")

axs[1, 2].set_title(r"$\beta_1$")
axs[1, 2].set_xlabel("Lag")
axs[1, 2].set_ylabel("Correlation")

axs[2, 0].set_title(r"$\beta_2$")
axs[2, 0].set_xlabel("Lag")
axs[2, 0].set_ylabel("Correlation")

plt.tight_layout()
plt.savefig("./out/stan_out/autocorrelation_simulation.png", dpi=300)
# plt.show()
plt.close()
# import ipdb

# ipdb.set_trace()


axs = az.plot_posterior(
    fit, hdi_prob=0.95, point_estimate=None, textsize=20, figsize=(15, 15)
)
axs[0, 0].set_title(r"$\sigma^2$")
axs[0, 0].axvline(sigma2_true, linestyle="--", color="red")

axs[0, 1].set_title(r"$b$")
axs[0, 1].axvline(b_true, linestyle="--", color="red")

axs[0, 2].set_title(r"$a$")
axs[0, 2].axvline(a_true, linestyle="--", color="red")

axs[1, 0].set_title(r"$\tau^2_1$")
axs[1, 0].axvline(noise_variance[0], linestyle="--", color="red")

axs[1, 1].set_title(r"$\tau^2_2$")
axs[1, 1].axvline(noise_variance[1], linestyle="--", color="red")

axs[1, 2].set_title(r"$\beta_1$")
axs[1, 2].axvline(mean_intercepts[0], linestyle="--", color="red")

axs[2, 0].set_title(r"$\beta_2$")
axs[2, 0].axvline(mean_intercepts[1], linestyle="--", color="red")

plt.savefig("./out/stan_out/posterior_distributions_simulation.png", dpi=300)
plt.show()
plt.close()


import ipdb

ipdb.set_trace()
