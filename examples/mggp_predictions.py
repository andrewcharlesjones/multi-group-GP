import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from multigroupGP import GP, MultiGroupRBF, MultiGroupMatern12
import numpy as onp

import matplotlib

font = {"size": 15}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

key = random.PRNGKey(1)

## Generate data
noise_variance = 0.05
n0, n1 = 100, 100
p = 1
n = n0 + n1
X0 = random.uniform(key, minval=-10, maxval=10, shape=(n0, 1))
X1 = random.uniform(key, minval=-10, maxval=10, shape=(n1, 1))
X = jnp.concatenate([X0, X1], axis=0)

group_dist_mat = onp.ones((2, 2)) - onp.eye(2)

X_groups = onp.concatenate([onp.zeros(n0), onp.ones(n1)]).astype(int)

kernel_true = MultiGroupRBF(amplitude=1.0, lengthscale=1.0, group_diff_param=1.0)
K_XX = (
    kernel_true(
        x1=X,
        groups1=X_groups.astype(float),
        group_distances=group_dist_mat,
    )
    + noise_variance * jnp.eye(n)
)
Y = random.multivariate_normal(random.PRNGKey(12), jnp.zeros(n), K_XX)

## Set up GP
kernel = MultiGroupRBF()
# kernel = MultiGroupMatern12()
mggp = GP(kernel=kernel, key=key, is_mggp=True)
mggp.fit(
    X,
    Y,
    groups=X_groups,
    group_distances=group_dist_mat,
    group_specific_noise_terms=True,
)

## Predict
ntest = 200
Xtest_onegroup = jnp.linspace(-10, 10, ntest)[:, None]
Xtest = jnp.concatenate([Xtest_onegroup, Xtest_onegroup], axis=0)
Xtest_groups = onp.concatenate([onp.zeros(ntest), onp.ones(ntest)]).astype(int)
preds_mean = mggp.predict(Xtest, groups_test=Xtest_groups)

## Plot
plt.figure(figsize=(10, 5))
plt.scatter(X[:n0], Y[:n0], color="black", label="Data, group 1")
plt.scatter(X[n0:], Y[n0:], color="gray", label="Data, group 2")
plt.plot(Xtest[:ntest], preds_mean[:ntest], color="red", label="Predictions, group 1")
plt.plot(
    Xtest[ntest:], preds_mean[ntest:], color="orange", label="Predictions, group 2"
)
plt.xlabel(r"$X$")
plt.ylabel(r"$y$")
plt.legend()
plt.tight_layout()
plt.savefig("./images/mggp_preds.png")
plt.show()
import ipdb

ipdb.set_trace()
