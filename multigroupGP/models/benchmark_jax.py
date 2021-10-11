from jax_mggp import GP as JAXGP, rbf_kernel as rbf_kernel_jax
import jax.numpy as jnp
import numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import time
import sys
from jax import grad as jax_grad
from autograd import grad as og_grad
from sklearn.gaussian_process.kernels import RBF
import autograd.numpy as anp
from jax import jit

sys.path.append("..")
from kernels.kernels import rbf_covariance as rbf_kernel_vanilla

from gaussian_process import GP

key = random.PRNGKey(0)

true_params_jax = jnp.zeros(4)
true_params_np = anp.zeros(4)

gp_jax = JAXGP(kernel=rbf_kernel_jax, key=key)
grad_fun_jax = jit(jax_grad(gp_jax.log_marginal_likelihood))

gp_vanilla = GP(kernel=rbf_kernel_vanilla)
grad_fun_vanilla = og_grad(gp_vanilla.log_marginal_likelihood)


jax_times = []
vanilla_times = []
for _ in range(20):
    ## Make data
    n = 50
    noise_variance = 0.01
    X = jnp.linspace(-5, 5, n)[:, None]
    K_XX = RBF()(X, X) + noise_variance * jnp.eye(n)
    Y = random.multivariate_normal(key, jnp.zeros(n), K_XX)
    X_vanilla, Y_vanilla = anp.asarray(X), anp.asarray(Y)

    ## JAX GP
    t0 = time.time()
    # grad_fun_jax(true_params_jax, X, Y)
    gp_jax.log_marginal_likelihood(true_params_jax, X, Y)
    t1 = time.time()
    jax_time = t1 - t0

    ## Vanilla GP
    t0 = time.time()
    # grad_fun_vanilla(true_params_np, X_vanilla, Y_vanilla)
    gp_vanilla.log_marginal_likelihood(true_params_np, X, Y)
    t1 = time.time()
    vanilla_time = t1 - t0

    jax_times.append(jax_time)
    vanilla_times.append(vanilla_time)

plt.boxplot([jax_times[1:], vanilla_times[1:]])
plt.show()
import ipdb

ipdb.set_trace()
# gp.fit(X, Y)
# preds_mean, _ = gp.predict(X)

# plt.scatter(X, Y)
# plt.scatter(X, preds_mean)
# plt.show()
