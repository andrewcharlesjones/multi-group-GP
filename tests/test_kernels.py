import jax.numpy as jnp
import numpy as onp
from multigroupGP import GP, RBF, Matern12, MultiGroupRBF, MultiGroupMatern12


def test_sole_input():
	n = 100
	X = onp.random.uniform(-5, 5, size=(n, 1))
	params = [1.0, 1.0]

	assert jnp.array_equal(
		RBF()(params, X, log_params=False),
		RBF()(params, X, X, log_params=False)
	)

	assert jnp.array_equal(
		Matern12()(params, X, log_params=False),
		Matern12()(params, X, X, log_params=False)
	)

	params = [1.0, 1.0, 1.0]
	groups = onp.random.choice([0, 1], size=n)

	assert jnp.array_equal(
		MultiGroupRBF()(params, x1=X, groups1=groups, log_params=False),
		MultiGroupRBF()(params, x1=X, x2=X, groups1=groups, groups2=groups, log_params=False)
	)

	params = [1.0, 1.0, 1.0, 1.0]

	assert jnp.array_equal(
		MultiGroupMatern12()(params, x1=X, groups1=groups, log_params=False),
		MultiGroupMatern12()(params, x1=X, x2=X, groups1=groups, groups2=groups, log_params=False)
	)

def test_union_mggp_relationship():
	n = 100
	X = onp.random.uniform(-5, 5, size=(n, 1))

	params = [1.0, 1.0]
	K_rbf = RBF()(params, X, log_params=False)

	params = [1.0, 0.0, 1.0]
	groups = onp.random.choice([0, 1], size=n)
	K_mgrbf = MultiGroupRBF()(params, x1=X, groups1=groups, log_params=False)

	assert jnp.array_equal(K_rbf, K_mgrbf)
