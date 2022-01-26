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
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF
import numpy as onp
import seaborn as sns
import warnings


def hgp_kernel(
    kernel_params,
    x1,
    groups1,
    within_group_kernel,
    between_group_kernel,
    x2=None,
    groups2=None,
):
    if x2 is None:
        x2 = x1
        groups2 = groups1

    same_group_mask = (
        onp.expand_dims(groups1, 1) == onp.expand_dims(groups2, 0)
    ).astype(int)

    n_params_total = len(kernel_params)
    within_group_params = kernel_params[n_params_total // 2 :]
    between_group_params = kernel_params[: n_params_total // 2]

    K_within = within_group_kernel(within_group_params, x1, x2)
    K_between = between_group_kernel(between_group_params, x1, x2)

    K = K_between + same_group_mask * K_within

    return K


def make_gp_funs(
    cov_func, num_cov_params, is_hgp=False, is_mggp=False, n_noise_terms=1
):
    def unpack_kernel_params(params):
        """Splits list of parameters into mean, noise variance, and kernel parameters.

        Args:
            params (iterable): List/array of model and kernel parameters.

        Returns:
            triple: mean (scalar), kernel parameters (array), and noise scale.
        """
        mean = params[0]
        noise_scale = jnp.exp(params[1 : n_noise_terms + 1]) + 0.0001
        cov_params = jnp.array(params[n_noise_terms + 1 :])
        return mean, cov_params, noise_scale

    def predict(params, x, y, xstar, return_cov=False, **kwargs):
        """Prediction function for the (MG)GP

        Args:
            params (iterable): List/array of kernel parameters.
            x (array): n x p array of training data.
            y (array): n-vector of training responses.
            xstar (array): m x p array of testing data.
            return_cov (bool, optional): Whether to return the covariance of the predictive distribution.
            **kwargs: groups and group distances (for MGGP)

        Returns:
            TYPE: Predictive mean or (predictive mean, predictive covariance)
        """
        mean, cov_params, noise_scales = unpack_kernel_params(params)

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

        cov_f_f = cov_func(params=cov_params, x1=xstar, x2=xstar, **extra_args)

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

        cov_y_f = cov_func(params=cov_params, x1=x, x2=xstar, **extra_args)

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

        if len(noise_scales) == 1:
            noise_scale = noise_scales[0]
        else:
            noise_scale = noise_scales[kwargs["x_groups"]]

        cov_y_y = cov_func(
            params=cov_params, x1=x, x2=x, **extra_args
        ) + noise_scale * jnp.eye(len(y))

        chol = scipy.linalg.cholesky(cov_y_y, lower=True)
        Kinv_y = scipy.linalg.solve_triangular(
            chol.T, scipy.linalg.solve_triangular(chol, y - mean, lower=True)
        )
        pred_mean = mean + jnp.dot(cov_y_f.T, Kinv_y)
        if return_cov:
            Kinv_Kyf = scipy.linalg.solve_triangular(chol.T, scipy.linalg.solve_triangular(chol, cov_y_f, lower=True))
            pred_cov = cov_f_f - jnp.dot(cov_y_f.T, Kinv_Kyf)
            return pred_mean, pred_cov
        else:
            return pred_mean

    def log_marginal_likelihood(params, x, y, **kwargs):

        mean, cov_params, noise_scales = unpack_kernel_params(params)
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

        if len(noise_scales) == 1:
            noise_scale = noise_scales[0]
        else:
            noise_scale = noise_scales[kwargs["groups"]]

        cov_y_y = cov_func(
            params=cov_params, x1=x, x2=x, **extra_args
        ) + noise_scale * jnp.eye(len(y))
        prior_mean = mean * jnp.ones(len(y))

        return scipy.stats.multivariate_normal.logpdf(y, prior_mean, cov_y_y)

    return (
        num_cov_params + 1 + n_noise_terms,
        predict,
        log_marginal_likelihood,
        unpack_kernel_params,
    )


class GP:
    def __init__(
        self,
        kernel,
        is_mggp=False,
        is_hgp=False,
        within_group_kernel=None,
        key=random.PRNGKey(0),
        group_specific_noise_terms=False,
    ):

        self.key = key
        if is_mggp and is_hgp:
            raise ValueError("GP cannot be both an MGGP and HGP.")

        self.is_mggp = is_mggp
        self.is_hgp = is_hgp
        self.group_specific_noise_terms = group_specific_noise_terms

        # if self.is_hgp:
        #     if within_group_kernel is None:
        #         raise ValueError("Must specify within-group kernel function for HGP.")
        #     self.kernel = lambda kernel_params, x1, x2, groups1, groups2: hgp_kernel(
        #         kernel_params=kernel_params,
        #         x1=x1,
        #         x2=x2,
        #         groups1=groups1,
        #         groups2=-groups2,
        #         within_group_kernel=within_group_kernel,
        #         between_group_kernel=kernel,
        #     )
        # else:
        #     self.kernel = kernel
        self.kernel = kernel

        if self.is_mggp:
            if self.group_specific_noise_terms:
                n_groups = len(onp.unique(groups))
                self.n_noise_terms = n_groups
            else:
                self.n_noise_terms = 1
        else:
            self.n_noise_terms = 1

        # Build model
        (
            num_params,
            predict,
            log_marginal_likelihood,
            unpack_kernel_params,
        ) = make_gp_funs(
            self.kernel,
            num_cov_params=self.kernel.num_cov_params,
            is_mggp=self.is_mggp,
            is_hgp=self.is_hgp,
            n_noise_terms=self.n_noise_terms,
        )

        self.num_params = num_params
        self.predict_fn = predict
        self.log_marginal_likelihood = log_marginal_likelihood

    def callback(self, params):
        pass

    def set_up_objective(
        self, X, y, groups=None, group_distances=None, n_noise_terms=1
    ):
        self.X = X
        self.y = y
        self.groups = groups
        self.group_distances = group_distances

        # # Build model
        # (
        #     num_params,
        #     predict,
        #     log_marginal_likelihood,
        #     unpack_kernel_params,
        # ) = make_gp_funs(
        #     self.kernel,
        #     num_cov_params=self.kernel.num_cov_params,
        #     is_mggp=self.is_mggp,
        #     is_hgp=self.is_hgp,
        #     n_noise_terms=n_noise_terms,
        # )

        # self.num_params = num_params
        # self.predict_fn = predict
        # self.log_marginal_likelihood = log_marginal_likelihood

        self.objective = lambda params: -self.log_marginal_likelihood(
            params,
            self.X,
            self.y,
            groups=self.groups,
            group_distances=self.group_distances,
        )

        # Initialize covariance parameters
        self.init_params = 0.1 * random.normal(self.key, shape=(self.num_params,))

    def maximize_LL(self, tol, verbose, print_every, max_steps, learning_rate):

        params = 0.1 * onp.random.normal(size=(self.num_params))

        # initialize optimizer
        # lr = 0.01  # Learning rate
        opt_init, opt_update, get_params = optimizers.adam(step_size=learning_rate)
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
                print(
                    "Step: {:<15} Log marginal lik.: {}".format(
                        step_num, onp.round(-1 * onp.asarray(curr_mll), 2)
                    )
                )

        fitted_params = get_params(opt_state)
        cov_params = fitted_params[-self.kernel.num_cov_params :]
        self.params = fitted_params
        self.kernel.store_params(jnp.exp(cov_params))
        self.kernel.is_fitted = True

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
        group_specific_noise_terms=False,
        learning_rate=1e-2,
    ):
        if self.is_mggp:
            if group_distances is None:
                group_distances = 1 - onp.eye(len(onp.unique(groups)))
                onp.fill_diagonal(group_distances, 0)
            if not onp.all(onp.diag(group_distances) == 0):
                raise ValueError("Distance from a group to itself cannot be nonzero.")

        self.set_up_objective(
            X,
            y,
            groups=groups,
            group_distances=group_distances,
            n_noise_terms=self.n_noise_terms,
        )
        self.maximize_LL(
            tol=tol, verbose=verbose, print_every=print_every, max_steps=max_steps, learning_rate=learning_rate
        )

    def predict(self, X_test, groups_test=None, return_cov=False):

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
    import jax.numpy as jnp
    import numpy as onp
    import jax.random as random
    import matplotlib.pyplot as plt
    from multigroupGP import GP, RBF
    from sklearn.gaussian_process.kernels import RBF as RBF_sklearn
    from scipy.stats import multivariate_normal as mvn

    key = random.PRNGKey(1)
    n = 1000
    noise_variance = 0.01
    X = onp.linspace(-5, 5, n).reshape(-1, 1)
    K_XX = RBF_sklearn()(X, X)
    y = mvn.rvs(mean=onp.zeros(n), cov=K_XX + 1e-8 * onp.eye(n))
    y += onp.random.normal(scale=onp.sqrt(noise_variance), size=y.shape[0])

    kernel = RBF()
    gp = GP(kernel=kernel, key=key)
    gp.fit(X, y)
    # preds_mean = gp.predict(Xtest)
