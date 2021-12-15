import matplotlib
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pcpca import PCPCA
import numpy as np
import jax.numpy as jnp
import stan
from hashlib import md5
from os.path import join as pjoin
import pickle
import os
from sklearn.model_selection import train_test_split
from multigroupGP import GP, MultiGroupRBF
import arviz as az
import pandas as pd
import seaborn as sns

MODEL_DIR = "../../multigroupGP/models"


font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

N_MCMC_ITER = 200
N_WARMUP_ITER = 200
N_CHAINS = 4

frac_train = 0.75
n0, n1 = 20, 20
ntotal = n0 + n1
noise_variance = 0.05
p = 1
n_groups = 2
n_infinite = 200

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

X_full = np.concatenate(
    [
        np.expand_dims(np.linspace(-10, 10, n_infinite // 2), 1),
        np.expand_dims(np.linspace(-10, 10, n_infinite // 2), 1),
    ]
)

sigma2_true = 1.0
a_true = 1.0
b_true = 1.0
mean_intercepts = np.array([0, 0])

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
# plt.scatter(X_full[groups_full == 0], Y_full_noiseless[groups_full == 0])
# plt.scatter(X_full[groups_full == 1], Y_full_noiseless[groups_full == 1])
# plt.show()

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


kernel = MultiGroupRBF()
mggp = GP(kernel=kernel, is_mggp=True)
mggp.fit(Xtrain, Ytrain, groups=groups_train)
preds_mean, pred_cov = mggp.predict(X_full, groups_test=groups_full, return_cov=True)
preds_stddev = jnp.sqrt(jnp.diagonal(pred_cov))

plt.scatter(
    Xtrain[groups_train == 0],
    Ytrain[groups_train == 0],
    color="blue",
    s=150,
)
plt.scatter(
    Xtrain[groups_train == 1],
    Ytrain[groups_train == 1],
    color="red",
    s=150,
)


colors = ["blue", "red"]
for groupnum in [0, 1]:
    curr_idx = np.where(groups_full == groupnum)[0]
    plt.plot(X_full[curr_idx], preds_mean[curr_idx], c=colors[groupnum])

# colors = ["blue", "red"]
# for groupnum in [0, 1]:
#     curr_idx = np.where(groups_test == groupnum)[0]


#     ## Plot Y
#     # curr_Ytest_samples = Ytest_samples[curr_idx, :]
#     curr_Xtest = Xtest[curr_idx, :]
#     # preds_mean = curr_Ytest_samples.mean(1)
#     # preds_stddev = curr_Ytest_samples.std(1)
#     preds_upper = preds_mean + 2 * preds_stddev
#     preds_lower = preds_mean - 2 * preds_stddev
#     sorted_idx = np.argsort(curr_Xtest.squeeze())
#     plt.plot(
#         curr_Xtest.squeeze()[sorted_idx],
#         preds_mean[sorted_idx],
#         c=colors[groupnum],
#         alpha=0.5,
#         label=r"$Y(x; c_" + str(groupnum + 1) + ")$",
#         linestyle="-",
#         linewidth=7,
#     )
#     plt.fill_between(
#         curr_Xtest.squeeze()[sorted_idx],
#         preds_lower[sorted_idx],
#         preds_upper[sorted_idx],
#         alpha=0.2,
#         color=colors[groupnum],
#     )

plt.show()

import ipdb

ipdb.set_trace()


# Xtest = X_full
# Ytest = Y_full
# groups_test = groups_full
# ntrain, ntest = Xtrain.shape[0], Xtest.shape[0]
# # import ipdb; ipdb.set_trace()
# data = {
#     "N1": ntrain,
#     "N2": ntest,
#     "x1": Xtrain.squeeze(),
#     "x2": Xtest.squeeze(),
#     "design1": groups_oh_train,
#     "design2": groups_oh_full,
#     "y1": Ytrain,
#     "groups1": groups_train,
#     "groups2": groups_test,
# }

# ## Load model
# with open(pjoin(MODEL_DIR, "mggp.stan"), "r") as file:
#     model_code = file.read()

# # Set up model
# posterior = stan.build(model_code, data=data)

# # Start sampling
# fit = posterior.sample(
#     num_chains=N_CHAINS, num_warmup=N_WARMUP_ITER, num_samples=N_MCMC_ITER
# )

# arviz_summary = az.summary(fit)

# plt.figure(figsize=(12, 5))

# plt.scatter(
#     Xtrain[groups_train == 0],
#     Ytrain[groups_train == 0],
#     color="blue",
#     s=150,
# )
# plt.scatter(
#     Xtrain[groups_train == 1],
#     Ytrain[groups_train == 1],
#     color="red",
#     s=150,
# )

# Ytest_samples = fit["y2"]
# Ftest_samples = fit["f"][ntrain:, :]
# # import ipdb; ipdb.set_trace()

# colors = ["blue", "red"]
# for groupnum in [0, 1]:
#     curr_idx = np.where(groups_test == groupnum)[0]


#     ## Plot Y
#     curr_Ytest_samples = Ytest_samples[curr_idx, :]
#     curr_Xtest = Xtest[curr_idx, :]
#     preds_mean = curr_Ytest_samples.mean(1)
#     preds_stddev = curr_Ytest_samples.std(1)
#     preds_upper = preds_mean + 2 * preds_stddev
#     preds_lower = preds_mean - 2 * preds_stddev
#     sorted_idx = np.argsort(curr_Xtest.squeeze())
#     plt.plot(
#         curr_Xtest.squeeze()[sorted_idx],
#         preds_mean[sorted_idx],
#         c=colors[groupnum],
#         alpha=0.5,
#         label=r"$Y(x; c_" + str(groupnum + 1) + ")$",
#         linestyle="-",
#         linewidth=7,
#     )
#     plt.fill_between(
#         curr_Xtest.squeeze()[sorted_idx],
#         preds_lower[sorted_idx],
#         preds_upper[sorted_idx],
#         alpha=0.2,
#         color=colors[groupnum],
#     )


# plt.legend(bbox_to_anchor=(1.1, 1.05))
# plt.tight_layout()
# plt.savefig("../../plots/mggp_predictive_samples.png")
# plt.show()
# plt.close()


# import ipdb

# ipdb.set_trace()
