import matplotlib
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pcpca import PCPCA
import numpy as np
import pystan
from hashlib import md5
from os.path import join as pjoin
import pickle
import os

# from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

import sys

sys.path.append("../kernels")
from kernels import multigroup_rbf_covariance

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

N_MCMC_ITER = 60

frac_train = 0.75
n0, n1 = 40, 40
ntotal = n0 + n1
noise_variance = 0.01
p = 1
n_groups = 2
n_infinite = 80

X0_group_one_hot = np.zeros(n_groups)
X0_group_one_hot[0] = 1
X0_groups = np.repeat([X0_group_one_hot], n_infinite//2, axis=0)
X1_group_one_hot = np.zeros(n_groups)
X1_group_one_hot[1] = 1
X1_groups = np.repeat([X1_group_one_hot], n_infinite//2, axis=0)
groups_oh_full = np.concatenate([X0_groups, X1_groups], axis=0)
groups_full = np.concatenate(
		[np.zeros(n_infinite//2),
		np.ones(n_infinite//2)]
	)

X_full = np.expand_dims(np.linspace(-5, 5, n_infinite), 1)
X_full = X_full[np.random.choice(np.arange(n_infinite), size=n_infinite, replace=True), :]

sigma2_true = 1.0
a_true = 1.0
b_true = 1.0

K_XX = multigroup_rbf_covariance(
	kernel_params=[np.log(sigma2_true), np.log(a_true), np.log(b_true)],
	X1=X_full,
	X2=X_full,
	groups1=groups_oh_full,
	groups2=groups_oh_full,
	group_distances=np.ones((2, 2)),
)

Y_full_noiseless = mvn(np.zeros(n_infinite), K_XX + np.eye(n_infinite) * 1e-5).rvs()
Y_full = Y_full_noiseless + np.random.normal(scale=np.sqrt(noise_variance), size=n_infinite)


group0_idx = np.random.choice(np.arange(n_infinite // 2), size=n0)
group1_idx = np.random.choice(np.arange(n_infinite // 2, n_infinite), size=n0)
X = np.concatenate(
		[X_full[group0_idx], X_full[group1_idx]]
	)
Y = np.concatenate(
		[Y_full[group0_idx], Y_full[group1_idx]]
	)
groups = np.concatenate(
		[groups_full[group0_idx], groups_full[group1_idx]]
	)
groups_oh = np.concatenate(
		[groups_oh_full[group0_idx], groups_oh_full[group1_idx]]
	)

Xtrain, Xtest, Ytrain, Ytest, groups_train, groups_test = train_test_split(
	X, Y, groups, test_size=1 - frac_train, random_state=42
)
Xtest = X_full
Ytest = Y_full
groups_test = groups_full
ntrain, ntest = Xtrain.shape[0], Xtest.shape[0]

data = {
	"N1": ntrain,
	"N2": ntest,
	"x1": Xtrain.squeeze(),
	"x2": Xtest.squeeze(),
	"y1": Ytrain,
	"groups1": groups_train,
	"groups2": groups_test,
}

## Load model
with open("gp.stan", "r") as file:
	model_code = file.read()
code_hash = md5(model_code.encode("ascii")).hexdigest()
cache_fn = pjoin("cached_models", "cached-model-{}.pkl".format(code_hash))

if os.path.isfile(cache_fn):
	print("Loading cached model...")
	sm = pickle.load(open(cache_fn, "rb"))
else:
	print("Saving model to cache...")
	sm = pystan.StanModel(model_code=model_code)
	with open(cache_fn, "wb") as f:
		pickle.dump(sm, f)

# Fit model
fit = sm.sampling(data=data, iter=N_MCMC_ITER)


plt.figure(figsize=(12, 5))

plt.scatter(Xtrain[groups_train == 0], Ytrain[groups_train == 0], color="blue", label="Group 1")
plt.scatter(Xtrain[groups_train == 1], Ytrain[groups_train == 1], color="red", label="Group 2")

Ytest_samples = fit.extract()["y2"]

## Group 1
colors = ["blue", "red"]
for groupnum in [0, 1]:
	curr_idx = np.where(groups_test == groupnum)[0]
	curr_Ytest_samples = Ytest_samples[:, curr_idx]
	curr_Xtest = Xtest[curr_idx, :]
	preds_mean = curr_Ytest_samples.mean(0)
	preds_stddev = curr_Ytest_samples.std(0)
	preds_upper = preds_mean + 2 * preds_stddev
	preds_lower = preds_mean - 2 * preds_stddev
	sorted_idx = np.argsort(curr_Xtest.squeeze())
	plt.plot(curr_Xtest.squeeze()[sorted_idx], preds_mean[sorted_idx], c=colors[groupnum], alpha=0.5)
	plt.fill_between(curr_Xtest.squeeze()[sorted_idx], preds_lower[sorted_idx], preds_upper[sorted_idx], alpha=0.3, color=colors[groupnum])


plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.tight_layout()
plt.savefig("../plots/mggp_predictive_samples.png")
# plt.show()
plt.close()
# import ipdb; ipdb.set_trace()

plt.figure(figsize=(21, 7))
plt.subplot(131)
plt.hist(fit.extract()["alpha"])
plt.xlabel(r"$a$")
plt.ylabel("Density")
plt.axvline(a_true, color="red", linestyle="--")

plt.subplot(132)
plt.hist(fit.extract()["length_scale"])
plt.xlabel(r"$b$")
plt.ylabel("Density")
plt.axvline(b_true, color="red", linestyle="--")

plt.subplot(133)
plt.hist(fit.extract()["output_variance"])
plt.xlabel(r"$\sigma^2$")
plt.ylabel("Density")
plt.axvline(sigma2_true, color="red", linestyle="--")
plt.savefig("../plots/mggp_hyperparameter_samples.png")
plt.show()

import ipdb

ipdb.set_trace()
