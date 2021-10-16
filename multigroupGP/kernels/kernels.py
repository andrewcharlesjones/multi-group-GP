from absl import app
from absl import flags
from functools import partial
from jax import grad, value_and_grad
from jax import jit
from jax import vmap
from jax.config import config
import jax.numpy as jnp
import jax.random as random
import jax.scipy as scipy
from jax.experimental import optimizers
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF
import numpy as onp
import seaborn as sns
import warnings
from sklearn.metrics import pairwise_distances


def softplus(x):
    return jnp.logaddexp(x, 0.0)


def cov_map(cov_func, xs, xs2=None):
    if xs2 is None:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T


def embed_distance_matrix(distance_matrix):
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
    n_groups = distance_matrix.shape[0]
    D2 = distance_matrix ** 2
    C = onp.eye(n_groups) - 1 / n_groups * onp.ones((n_groups, n_groups))
    B = -0.5 * C @ D2 @ C
    eigvals, E = onp.linalg.eigh(B)
    embedding = E @ onp.diag(onp.sqrt(eigvals + 1e-10))
    return embedding


##############################
############ RBF #############
##############################


def rbf_kernel_vectorized(x1, x2):
    return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2))


def rbf_kernel(kernel_params, x1, x2=None):
    try:
        assert len(kernel_params.primal) == 2
    except:
        assert len(kernel_params) == 2
    output_scale = jnp.exp(kernel_params[0])
    lengthscale = jnp.exp(kernel_params[1])
    x1 = x1 / lengthscale
    if x2 is not None:
        x2 = x2 / lengthscale
    cov = output_scale * cov_map(rbf_kernel_vectorized, x1, x2)
    return cov


##############################
###### Multi-group RBF #######
##############################


def cov_map_multigroup_rbf(
    cov_func, xs, group_embeddings1, group_diff_param, xs2=None, group_embeddings2=None
):
    if xs2 is None:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(
            lambda x, g1: vmap(
                lambda y, g2: cov_func(x, y, g1, g2, group_diff_param=group_diff_param)
            )(xs, group_embeddings1)
        )(xs2, group_embeddings2).T


def multigroup_rbf_kernel_vectorized(
    x1, x2, group_embeddings1, group_embeddings2, group_diff_param
):

    p = x1.shape[-1]
    dists = jnp.sum((x1 - x2) ** 2)
    group_dists = jnp.sum((group_embeddings1 - group_embeddings2) ** 2)

    cov = (
        1
        / (group_diff_param * group_dists + 1) ** (0.5 * p)
        * jnp.exp(-0.5 * dists / (group_diff_param * group_dists + 1))
    )
    return cov


def multigroup_rbf_kernel(
    kernel_params, x1, groups1, group_distances, x2=None, groups2=None
):
    try:
        assert len(kernel_params.primal) == 3
    except:
        assert len(kernel_params) == 3

    if not isinstance(groups1.flat[0], onp.integer):
        warnings.warn(
            "Casting group labels to integers. Make sure your group labels are ints to avoid undue casting!"
        )
        groups1 = groups1.astype(int)
        if groups2 is not None:
            groups2 = groups2.astype(int)

    output_scale = jnp.exp(kernel_params[0])
    group_diff_param = jnp.exp(kernel_params[1])
    lengthscale = jnp.exp(kernel_params[2])

    assert onp.all(onp.diag(group_distances) == 0)

    ## Embed group distance matrix in Euclidean space for convenience.
    embedding = embed_distance_matrix(group_distances)

    x1 = x1 / (lengthscale)
    group_embeddings1 = jnp.array([embedding[xx] for xx in groups1])
    if x2 is not None:
        x2 = x2 / (lengthscale)
        group_embeddings2 = jnp.array([embedding[xx] for xx in groups2])
    else:
        x2 = x1
        group_embeddings2 = group_embeddings1

    cov = cov_map_multigroup_rbf(
        multigroup_rbf_kernel_vectorized,
        xs=x1,
        xs2=x2,
        group_embeddings1=group_embeddings1,
        group_embeddings2=group_embeddings2,
        group_diff_param=group_diff_param,
    )
    return output_scale * cov.squeeze()


##############################
### Multi-group Matern 1/2 ###
##############################


def cov_map_multigroup_matern12(
    cov_func,
    xs,
    group_embeddings1,
    lengthscale,
    group_diff_param,
    dependency_scale,
    xs2=None,
    group_embeddings2=None,
):
    if xs2 is None:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(
            lambda x, g1: vmap(
                lambda y, g2: cov_func(
                    x,
                    y,
                    g1,
                    g2,
                    lengthscale=lengthscale,
                    group_diff_param=group_diff_param,
                    dependency_scale=dependency_scale,
                )
            )(xs, group_embeddings1)
        )(xs2, group_embeddings2).T


def cov_map_matern12(
    cov_func,
    xs,
    lengthscale,
    xs2=None,
):
    if xs2 is None:
        return vmap(
            lambda x: vmap(lambda y: cov_func(x, y, lengthscale=lengthscale))(xs)
        )(xs)
    else:
        return vmap(
            lambda x: vmap(lambda y: cov_func(x, y, lengthscale=lengthscale))(xs)
        )(xs2).T


def matern12_kernel_vectorized(x1, x2, lengthscale):
    return jnp.exp(-0.5 * jnp.sqrt(jnp.sum((x1 - x2) ** 2)) / lengthscale)


def matern12_kernel(kernel_params, x1, x2=None):
    try:
        assert len(kernel_params.primal) == 2
    except:
        assert len(kernel_params) == 2
    output_scale = jnp.exp(kernel_params[0])
    lengthscale = jnp.exp(kernel_params[1])
    cov = output_scale * cov_map_matern12(
        matern12_kernel_vectorized, xs=x1, xs2=x2, lengthscale=lengthscale
    )
    return cov.squeeze()


def multigroup_matern12_kernel_vectorized(
    x1,
    x2,
    group_embeddings1,
    group_embeddings2,
    lengthscale,
    group_diff_param,
    dependency_scale,
):

    p = x1.shape[-1]
    dists = jnp.sqrt(jnp.sum((x1 - x2) ** 2))
    group_dists = jnp.sum((group_embeddings1 - group_embeddings2) ** 2)

    pre_exp_term = dependency_scale ** (0.5 * p) / (
        (group_diff_param * group_dists + 1) ** 0.5
        * (group_diff_param * group_dists + dependency_scale) ** (0.5 * p)
    )

    exp_term = jnp.exp(
        -lengthscale
        * dists
        * (
            (group_diff_param * group_dists + 1)
            / (group_diff_param * group_dists + dependency_scale)
        )
        ** 0.5
    )

    cov = pre_exp_term * exp_term
    return cov


def multigroup_matern12_kernel(
    kernel_params, x1, groups1, group_distances, x2=None, groups2=None
):
    try:
        assert len(kernel_params.primal) == 4
    except:
        assert len(kernel_params) == 4

    if not isinstance(groups1.flat[0], onp.integer):
        warnings.warn(
            "Casting group labels to integers. Make sure your group labels are ints to avoid undue casting!"
        )
        groups1 = groups1.astype(int)
        if groups2 is not None:
            groups2 = groups2.astype(int)

    output_scale = jnp.exp(kernel_params[0])
    group_diff_param = jnp.exp(kernel_params[1])
    lengthscale = jnp.exp(kernel_params[2])
    dependency_scale = jnp.exp(kernel_params[3])

    assert onp.all(onp.diag(group_distances) == 0)

    ## Embed group distance matrix in Euclidean space for convenience.
    embedding = embed_distance_matrix(group_distances)

    group_embeddings1 = jnp.array([embedding[xx] for xx in groups1])
    if x2 is not None:
        group_embeddings2 = jnp.array([embedding[xx] for xx in groups2])
    else:
        x2 = x1
        group_embeddings2 = group_embeddings1

    cov = cov_map_multigroup_matern12(
        multigroup_matern12_kernel_vectorized,
        xs=x1,
        xs2=x2,
        group_embeddings1=group_embeddings1,
        group_embeddings2=group_embeddings2,
        lengthscale=lengthscale,
        group_diff_param=group_diff_param,
        dependency_scale=dependency_scale,
    )
    return output_scale * cov.squeeze()
