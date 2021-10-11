import jax.numpy as jnp
import numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import time
import sys
from jax import grad as jax_grad
from sklearn.gaussian_process.kernels import RBF
from jax import jit
from multigroupGP import GP, rbf_kernel


def test_jax_predictions():
    key = random.PRNGKey(0)

    true_params_jax = jnp.zeros(4)

    gp = GP(kernel=rbf_kernel, key=key)
    noise_variance = 1e-6

    n = 50
    X = jnp.linspace(-5, 5, n)[:, None]
    K_XX = RBF()(X, X) + noise_variance * jnp.eye(n)

    jax_times = []
    vanilla_times = []
    for _ in range(20):

        ## Make data
        Y = random.multivariate_normal(key, jnp.zeros(n), K_XX)

        ## JAX GP
        gp.fit(X, Y)
        preds = gp.predict(X, return_cov=False)
        assert np.allclose(preds, Y, atol=1e-2)
