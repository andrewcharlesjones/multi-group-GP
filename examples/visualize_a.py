import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from multigroupGP import GP, multigroup_rbf_kernel, multigroup_matern12_kernel
import numpy as onp

import matplotlib

font = {"size": 25}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

noise_variance = 1e-6
n0, n1 = 100, 100
p = 1
n = n0 + n1
group_dist_mat = onp.ones((2, 2))
onp.fill_diagonal(group_dist_mat, 0)
X_groups = onp.concatenate([onp.zeros(n0), onp.ones(n1)]).astype(int)

X0 = jnp.linspace(-10, 10, n0)[:, None]
X1 = jnp.linspace(-10, 10, n1)[:, None]
X = jnp.concatenate([X0, X1], axis=0)

a_list = [10**(int(x)) for x in range(-2, 2)]
n_a = len(a_list)

plt.figure(figsize=(15, 3 * n_a))

for ii, curr_a in enumerate(a_list):

    ## True covariance parameters
    true_params = [
        jnp.zeros(1),       # Amplitude
        jnp.log(curr_a),    # Group difference parameter (a)
        jnp.zeros(1),       # Lengthscale
    ]

    ## Generate data
    K_XX = (
        multigroup_rbf_kernel(
            true_params,
            x1=X,
            groups1=X_groups,
            group_distances=group_dist_mat,
        )
        + noise_variance * jnp.eye(n)
    )
    Y = random.multivariate_normal(random.PRNGKey(ii * 33), jnp.zeros(n), K_XX)

    ## Plot
    plt.subplot(2, onp.ceil(n_a / 2), ii + 1)
    plt.plot(X[:n0], Y[:n0], color="red", label="Group 1", linewidth=5)
    plt.plot(X[n0:], Y[n0:], color="blue", label="Group 2", linewidth=5)
    plt.xlabel(r"$X$")
    plt.ylabel(r"$y$")
    plt.title(r"$a = {}$".format(curr_a))
    plt.legend()
    plt.tight_layout()
plt.savefig("./images/a_visualization.png")
plt.show()
# import ipdb; ipdb.set_trace()
