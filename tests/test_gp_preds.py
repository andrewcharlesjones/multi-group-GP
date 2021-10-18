import jax.numpy as jnp
import numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import time
import sys
from jax import grad as jax_grad
from sklearn.gaussian_process.kernels import RBF as RBF_sklearn
from jax import jit
from multigroupGP import GP, RBF, MultiGroupRBF, Matern12, MultiGroupMatern12


def test_RBF_predictions():
    key = random.PRNGKey(0)

    gp = GP(kernel=RBF(), key=key)
    noise_variance = 1e-6

    n = 50
    X = jnp.linspace(-5, 5, n)[:, None]
    K_XX = RBF_sklearn()(X, X) + noise_variance * jnp.eye(n)

    for _ in range(10):

        ## Make data
        Y = random.multivariate_normal(key, jnp.zeros(n), K_XX)

        ## JAX GP
        gp.fit(X, Y)
        preds = gp.predict(X, return_cov=False)
        assert np.allclose(preds, Y, atol=1e-2)


def test_Matern12_predictions():
    key = random.PRNGKey(0)

    gp = GP(kernel=Matern12(), key=key)
    noise_variance = 1e-6

    n = 50
    X = jnp.linspace(-5, 5, n)[:, None]
    kernel_true = Matern12(amplitude=1.0, lengthscale=1.0)
    K_XX = kernel_true(x1=X, log_params=False) + noise_variance * jnp.eye(n)

    for _ in range(10):

        ## Make data
        Y = random.multivariate_normal(key, jnp.zeros(n), K_XX)

        ## Fit
        gp.fit(X, Y, tol=1e-10)
        preds = gp.predict(X, return_cov=False)
        # import ipdb; ipdb.set_trace()
        assert np.allclose(preds, Y, atol=1e-1)


def test_MGRBF_predictions():
    key = random.PRNGKey(3)

    n0, n1 = 100, 100
    p = 1
    n = n0 + n1
    X0 = random.uniform(key, minval=-10, maxval=10, shape=(n0, 1))
    X1 = random.uniform(key, minval=-10, maxval=10, shape=(n1, 1))
    X = jnp.concatenate([X0, X1], axis=0)
    noise_variance = 1e-5

    group_dist_mat = np.ones((2, 2)) - np.eye(2)

    X_groups = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)

    kernel_true = MultiGroupRBF(amplitude=1.0, lengthscale=1.0, group_diff_param=1.0)
    K_XX = (
        kernel_true(
            x1=X,
            groups1=X_groups,
            group_distances=group_dist_mat,
            log_params=False,
        )
        + noise_variance * jnp.eye(n)
    )

    for _ in range(10):

        ## Make data
        Y = random.multivariate_normal(random.PRNGKey(12), jnp.zeros(n), K_XX)

        kernel = MultiGroupRBF()
        mggp = GP(kernel=kernel, key=key, is_mggp=True)
        mggp.fit(
            X,
            Y,
            groups=X_groups,
            group_distances=group_dist_mat,
            group_specific_noise_terms=False,
            tol=1e-4,
        )
        preds = mggp.predict(X, X_groups)

        assert np.allclose(preds, Y, atol=1e-1)


def test_MGMatern12_predictions():
    key = random.PRNGKey(3)

    n0, n1 = 50, 50
    p = 1
    n = n0 + n1
    X0 = random.uniform(key, minval=-10, maxval=10, shape=(n0, 1))
    X1 = random.uniform(key, minval=-10, maxval=10, shape=(n1, 1))
    X = jnp.concatenate([X0, X1], axis=0)
    noise_variance = 1e-8

    group_dist_mat = np.ones((2, 2)) - np.eye(2)

    X_groups = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)

    kernel_true = MultiGroupMatern12(
        amplitude=1.0, lengthscale=1.0, group_diff_param=1.0, dependency_scale=1.0
    )
    K_XX = (
        kernel_true(
            x1=X,
            groups1=X_groups,
            group_distances=group_dist_mat,
            log_params=False,
        )
        + noise_variance * jnp.eye(n)
    )

    for _ in range(10):

        ## Make data
        Y = random.multivariate_normal(random.PRNGKey(12), jnp.zeros(n), K_XX)

        kernel = MultiGroupMatern12()
        mggp = GP(kernel=kernel, key=key, is_mggp=True)
        mggp.fit(
            X,
            Y,
            groups=X_groups,
            group_distances=group_dist_mat,
            group_specific_noise_terms=False,
            tol=1e-10,
        )
        preds = mggp.predict(X, X_groups)
        # import ipdb; ipdb.set_trace()
        assert np.allclose(preds, Y, atol=1e-1)


if __name__ == "__main__":
    test_Matern12_predictions()
