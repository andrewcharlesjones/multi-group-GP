import jax.numpy as jnp
import numpy as onp


def softplus(x):
    return jnp.logaddexp(x, 0.0)


def embed_distance_matrix(distance_matrix):
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
    n_groups = distance_matrix.shape[0]
    D2 = distance_matrix ** 2
    C = jnp.eye(n_groups) - 1 / n_groups * jnp.ones((n_groups, n_groups))
    B = -0.5 * C @ D2 @ C
    eigvals, E = jnp.linalg.eigh(B)
    embedding = E @ jnp.diag(jnp.sqrt(eigvals + 1e-10))
    return embedding
