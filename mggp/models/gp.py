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
import sys

sys.path.append("../kernels")
from kernels import multigroup_rbf_kernel, rbf_kernel, hgp_kernel


def make_gp_funs(cov_func, num_cov_params, is_hgp=False, is_mggp=False):
    def unpack_kernel_params(params):
        mean = params[0]
        noise_scale = jnp.exp(params[1]) + 0.0001
        cov_params = jnp.array(params[2:])
        return mean, cov_params, noise_scale

    def predict(params, x, y, xstar, return_cov=False, **kwargs):
        mean, cov_params, noise_scale = unpack_kernel_params(params)

        if is_hgp:
            extra_args = {
                "groups1": kwargs["xstar_groups"],
                "groups2": kwargs["xstar_groups"],
            }
        elif is_mggp:
            extra_args = {
                "groups1": kwargs["xstar_groups"],
                "groups2": kwargs["xstar_groups"],
                "group_distances": kwargs["group_distances"],
            }
        else:
            extra_args = {}

        cov_f_f = cov_func(kernel_params=cov_params, x1=xstar, x2=xstar, **extra_args)

        if is_hgp:
            extra_args = {
                "groups1": kwargs["x_groups"],
                "groups2": kwargs["xstar_groups"],
            }
        elif is_mggp:
            extra_args = {
                "groups1": kwargs["x_groups"],
                "groups2": kwargs["xstar_groups"],
                "group_distances": kwargs["group_distances"],
            }

        cov_y_f = cov_func(kernel_params=cov_params, x1=x, x2=xstar, **extra_args)

        if is_hgp:
            extra_args = {
                "groups1": kwargs["x_groups"],
                "groups2": kwargs["x_groups"],
            }
        elif is_mggp:
            extra_args = {
                "groups1": kwargs["x_groups"],
                "groups2": kwargs["x_groups"],
                "group_distances": kwargs["group_distances"],
            }

        cov_y_y = cov_func(
            kernel_params=cov_params, x1=x, x2=x, **extra_args
        ) + noise_scale * jnp.eye(len(y))

        chol = scipy.linalg.cholesky(cov_y_y, lower=True)
        Kinv_y = scipy.linalg.solve_triangular(
            chol.T, scipy.linalg.solve_triangular(chol, y - mean, lower=True)
        )
        pred_mean = mean + jnp.dot(cov_y_f.T, Kinv_y)
        if return_cov:
            pred_cov = cov_f_f - jnp.dot(Kinv_Kyf.T, cov_y_f)
            return pred_mean, preds_cov
        else:
            return pred_mean

    def log_marginal_likelihood(params, x, y, **kwargs):

        mean, cov_params, noise_scale = unpack_kernel_params(params)
        if is_hgp:
            extra_args = {"groups1": kwargs["groups"], "groups2": kwargs["groups"]}
        elif is_mggp:
            extra_args = {
                "groups1": kwargs["groups"],
                "groups2": kwargs["groups"],
                "group_distances": kwargs["group_distances"],
            }
        else:
            extra_args = {}

        cov_y_y = cov_func(
            kernel_params=cov_params, x1=x, x2=x, **extra_args
        ) + noise_scale * jnp.eye(len(y))
        prior_mean = mean * jnp.ones(len(y))

        return scipy.stats.multivariate_normal.logpdf(y, prior_mean, cov_y_y)

    return num_cov_params + 2, predict, log_marginal_likelihood, unpack_kernel_params


class GP:
    def __init__(self, kernel, is_mggp=False, is_hgp=False, within_group_kernel=None, key=random.PRNGKey(0)):
        
        self.key = key
        if is_mggp and is_hgp:
            raise ValueError("GP cannot be both an MGGP and HGP.")

        self.is_mggp = is_mggp
        self.is_hgp = is_hgp

        if self.is_hgp:
            if within_group_kernel is None:
                raise ValueError("Must specify within-group kernel function for HGP.")
            self.kernel = lambda kernel_params, x1, x2, groups1, groups2: hgp_kernel(
                kernel_params=kernel_params, x1=x1, x2=x2, groups1=groups1, groups2=-groups2, within_group_kernel=within_group_kernel, between_group_kernel=kernel
            )
        else:
            self.kernel = kernel
        

        if is_mggp:
            num_cov_params = 3
        elif is_hgp:
            num_cov_params = 4
        else:
            num_cov_params = 2
        self.num_cov_params = num_cov_params

        # Build model
        (
            num_params,
            predict,
            log_marginal_likelihood,
            unpack_kernel_params,
        ) = make_gp_funs(self.kernel, num_cov_params=num_cov_params, is_mggp=is_mggp, is_hgp=is_hgp)

        self.num_params = num_params
        self.predict_fn = predict
        self.log_marginal_likelihood = log_marginal_likelihood

    def callback(self, params):
        pass

    def set_up_objective(self, X, y, groups=None, group_distances=None):
        self.X = X
        self.y = y
        self.groups = groups
        self.group_distances = group_distances

        self.objective = lambda params: -self.log_marginal_likelihood(
            params,
            self.X,
            self.y,
            groups=self.groups,
            group_distances=self.group_distances,
        )

        # Initialize covariance parameters
        self.init_params = 0.1 * random.normal(self.key, shape=(self.num_params,))

    def maximize_LL(self, tol, verbose, print_every, max_steps):


        params = 0.1 * onp.random.normal(size=(self.num_params))

        # initialize optimizer
        lr = 0.01  # Learning rate
        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
        opt_state = opt_init(params)

        @jit
        def step(step, opt_state):
            value, grads = value_and_grad(self.objective)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state

        last_mll = 1e5
        for step_num in range(int(max_steps)):
            curr_mll, opt_state = step(step_num, opt_state)
            if jnp.abs(last_mll - curr_mll) < tol:
                break
            last_mll = curr_mll
            if verbose and step_num % print_every == 0:
                print("Step: {:<15} MLL: {}".format(step_num, onp.round(-1 * onp.asarray(curr_mll), 2)))

        self.params_dict = self.pack_params(get_params(opt_state))
        self.params = get_params(opt_state)

    def pack_params(self, params, exp=True):
        if exp:
            transformation = lambda x: jnp.exp(x)
        else:
            transformation = lambda x: x

        param_dict = {
            "mean": params[0],
            "noise_variance": transformation(params[1]),
            "amplitude": transformation(params[2]),
        }
        if self.num_cov_params == 2:
            param_dict["lengthscale"] = transformation(params[3]),
        elif self.num_cov_params == 3:
            param_dict["group_diff_param"] = transformation(params[3]),
            param_dict["lengthscale"] = transformation(params[4]),
        elif self.num_cov_params == 3:
            param_dict["lengthscale"] = transformation(params[3]),
            param_dict["amplitude_within_group"] = transformation(params[4]),
            param_dict["lengthscale_within_group"] = transformation(params[5]),
        return param_dict

    def print_params(self):
        for key, val in self.params_dict.items():
            print("{:<15} {}".format(key, onp.round(1 * onp.asarray(val), 2)))

    def fit(
        self,
        X,
        y,
        groups=None,
        group_distances=None,
        tol=1e-3,
        verbose=True,
        print_every=100,
        max_steps=1e5,
    ):
        if self.is_mggp:
            if group_distances is None:
                group_distances = onp.eye(len(onp.unique(groups)))
                onp.fill_diagonal(group_distances, 0)
            if not onp.all(onp.diag(group_distances) == 0):
                raise ValueError("Distance from a group to itself cannot be nonzero.")

        self.set_up_objective(X, y, groups=groups, group_distances=group_distances)
        self.maximize_LL(tol=tol, verbose=verbose, print_every=print_every, max_steps=max_steps)

    def predict(self, X_test, groups_test, return_cov=False):

        preds = self.predict_fn(
            self.params,
            self.X,
            self.y,
            xstar=X_test,
            x_groups=self.groups,
            xstar_groups=groups_test,
            group_distances=self.group_distances,
            return_cov=return_cov,
        )
        return preds


if __name__ == "__main__":
    key = random.PRNGKey(0)

    true_params = [
        jnp.zeros(1),
        jnp.zeros(1),
        # jnp.zeros(1),
    ]

    noise_variance = 0.01
    n0, n1 = 100, 100
    p = 1
    n = n0 + n1
    X0 = onp.linspace(-10, 10, n0)[:, None]
    X1 = onp.linspace(-10, 10, n1)[:, None]
    X = onp.concatenate([X0, X1])

    group_dist_mat = onp.ones((2, 2))
    onp.fill_diagonal(group_dist_mat, 0)

    X_groups = onp.concatenate([onp.zeros(n0), onp.ones(n1)]).astype(int)

    # K_XX = (
    #     multigroup_rbf_kernel(
    #         true_params,
    #         x1=X,
    #         groups1=X_groups,
    #         group_distances=group_dist_mat,
    #     )
    #     + noise_variance * jnp.eye(n)
    # )
    K_XX = (
        hgp_kernel(
            jnp.concatenate(jnp.array([true_params, true_params])),
            x1=X,
            groups1=X_groups,
            within_group_kernel=rbf_kernel,
            between_group_kernel=rbf_kernel,
        )
        + noise_variance * jnp.eye(n)
    )
    Y = random.multivariate_normal(key, jnp.zeros(n), K_XX)

    xtest = jnp.linspace(-5, 5, 200)[:, None]
    # gp = GP(kernel=multigroup_rbf_kernel, key=key, is_mggp=True)
    # gp.fit(X, Y, groups=X_groups, group_distances=group_dist_mat)
    gp = GP(kernel=rbf_kernel, within_group_kernel=rbf_kernel, key=key, is_hgp=True)
    gp.fit(X, Y, groups=X_groups)
    gp.print_params()
    # gp.fit(X, Y)
    # gp = GP(kernel=rbf_kernel, key=key, is_mggp=False)
    # print(gp.params)
    preds_mean = gp.predict(X, groups_test=X_groups)

    plt.scatter(X[:n0], Y[:n0])
    plt.scatter(X[n0:], Y[n0:])
    plt.plot(X[:n0], preds_mean[:n0], color="red")
    plt.plot(X[n0:], preds_mean[n0:], color="orange")
    plt.show()
    import ipdb

    ipdb.set_trace()
