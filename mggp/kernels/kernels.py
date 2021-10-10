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



def softplus(x):
    return jnp.logaddexp(x, 0.0)


def cov_map(cov_func, xs, xs2=None):
    if xs2 is None:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T


def cov_map_multigroup(
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


def rbf_kernel_vectorized(x1, x2):
    return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2))


def rbf_kernel(kernel_params, x1, x2=None):
    assert len(kernel_params) == 2
    output_scale = jnp.exp(kernel_params[0])
    lengthscale = jnp.exp(kernel_params[1])
    x1 = x1 / lengthscale
    if x2 is not None:
        x2 = x2 / lengthscale
    cov = output_scale * cov_map(rbf_kernel_vectorized, x1, x2)
    return cov


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
    output_scale = jnp.exp(kernel_params[0])
    group_diff_param = jnp.exp(kernel_params[1])
    lengthscale = jnp.exp(kernel_params[2])

    assert onp.all(onp.diag(group_distances) == 0)

    ## Embed group distance matrix in Euclidean space for convenience.
    B = group_distances @ group_distances.T + onp.eye(group_distances.shape[0]) * 1e-8
    eigvals, Q = onp.linalg.eigh(B)
    Lambda_sqrt = onp.diag(onp.sqrt(eigvals))
    embedding = Q @ Lambda_sqrt    

    x1 = x1 / (lengthscale)
    group_embeddings1 = jnp.array([embedding[xx] for xx in groups1])
    if x2 is not None:
        x2 = x2 / (lengthscale)
        group_embeddings2 = jnp.array([embedding[xx] for xx in groups2])
    else:
        x2 = x1
        group_embeddings2 = group_embeddings1

    cov = cov_map_multigroup(
        multigroup_rbf_kernel_vectorized,
        xs=x1,
        xs2=x2,
        group_embeddings1=group_embeddings1,
        group_embeddings2=group_embeddings2,
        group_diff_param=group_diff_param,
    )
    return output_scale * cov.squeeze()


def hgp_kernel(
    kernel_params, x1, groups1, within_group_kernel, between_group_kernel, x2=None, groups2=None
):
    
    if x2 is None:
        x2 = x1
        groups2 = groups1

    same_group_mask = (onp.expand_dims(groups1, 1) == onp.expand_dims(groups2, 0)).astype(int)

    n_params_total = len(kernel_params)
    within_group_params = kernel_params[n_params_total // 2 :]
    between_group_params = kernel_params[: n_params_total // 2]

    K_within = within_group_kernel(within_group_params, x1, x2)
    K_between = between_group_kernel(between_group_params, x1, x2)

    K = K_between + same_group_mask * K_within

    return K
