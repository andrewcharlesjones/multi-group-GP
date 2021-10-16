# import jax.numpy as jnp
# import numpy as np
# import jax.random as random
# import matplotlib.pyplot as plt
# import time
# import sys
# from jax import grad as jax_grad
# from sklearn.gaussian_process.kernels import RBF
# from jax import jit

# sys.path.append("..")
# from kernels.kernels import rbf_covariance as rbf_kernel_vanilla
# from models.gp import GP as JAXGP, rbf_kernel as rbf_kernel_jax
# from models.gp_autograd import GP as AGGP


# def test_jax_speed():
#     key = random.PRNGKey(0)

#     true_params_jax = jnp.zeros(4)
#     true_params = np.asarray(true_params_jax)

#     gp_jax = JAXGP(kernel=rbf_kernel_jax, key=key, is_mggp=False)
#     gp_ag = AGGP(kernel=rbf_kernel_vanilla)
#     noise_variance = 1e-6

#     n = 50
#     X = np.linspace(-5, 5, n)[:, None]
#     K_XX = RBF()(X, X) + noise_variance * np.eye(n)

#     jax_times = []
#     autograd_times = []
#     for _ in range(5):

#         ## Make data
#         Y = random.multivariate_normal(key, jnp.zeros(n), K_XX)

#         ## JAX GP
#         gp_jax.set_up_objective(X, Y)
#         t0 = time.time()
#         gp_jax.maximize_LL()
#         gp_jax.log_marginal_likelihood(true_params_jax, X, Y)
#         t1 = time.time()
#         jax_times.append(t1 - t0)

#         ## Autograd GP
#         t0 = time.time()
#         # gp_ag.fit(X, Y)
#         gp_ag.log_marginal_likelihood(true_params, X, Y)
#         t1 = time.time()
#         autograd_times.append(t1 - t0)

#     # assert np.all(np.array(jax_times) < np.array(autograd_times))

#     import ipdb

#     ipdb.set_trace()


# if __name__ == "__main__":
#     test_jax_speed()
