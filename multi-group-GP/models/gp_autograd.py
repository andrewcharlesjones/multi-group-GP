from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import value_and_grad
from autograd.scipy.linalg import solve_triangular, cholesky
from scipy.optimize import minimize

from scipy.stats import multivariate_normal as mvnpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.gaussian_process.kernels import RBF


import sys

sys.path.append("../kernels")
# from kernels import hgp_kernel

import matplotlib

font = {"size": 15}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


def make_gp_funs(cov_func, num_cov_params, is_hgp=False, is_mggp=False):
    """Functions that perform Gaussian process regression.
    cov_func has signature (cov_params, x, x')"""

    def unpack_kernel_params(params):
        mean = params[0]
        cov_params = params[2:]
        noise_scale = np.exp(params[1]) + 0.0001
        return mean, cov_params, noise_scale

    def predict(params, x, y, xstar, return_cov=False, **kwargs):
        """Returns the predictive mean and covariance at locations xstar,
        of the latent function value f (without observation noise)."""
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

        cov_f_f = cov_func(cov_params, xstar, xstar, **extra_args)

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

        cov_y_f = cov_func(cov_params, x, xstar, **extra_args)

        if is_hgp:
            extra_args = {"groups1": kwargs["x_groups"], "groups2": kwargs["x_groups"]}
        elif is_mggp:
            extra_args = {
                "groups1": kwargs["x_groups"],
                "groups2": kwargs["x_groups"],
                "group_distances": kwargs["group_distances"],
            }

        cov_y_y = cov_func(cov_params, x, x, **extra_args) + noise_scale * np.eye(
            len(y)
        )

        chol = cholesky(cov_y_y, lower=True)
        Kinv_y = solve_triangular(chol.T, solve_triangular(chol, y - mean, lower=True))
        pred_mean = mean + np.dot(cov_y_f.T, Kinv_y)

        # Kinv_Kyf = solve(cov_y_y, cov_y_f)
        # pred_mean = mean + np.dot(solve_triangular(chol, cov_y_f, lower=True).T, y - mean)
        # pred_mean = mean + np.dot(Kinv_Kyf.T, y - mean)
        # pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        if return_cov:
            pred_cov = cov_f_f - np.dot(Kinv_Kyf.T, cov_y_f)
            return pred_mean, pred_cov
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
        # import ipdb; ipdb.set_trace()
        cov_y_y = cov_func(cov_params, x, x, **extra_args) + noise_scale * np.eye(
            len(y)
        )
        prior_mean = mean * np.ones(len(y))
        return mvn.logpdf(y, prior_mean, cov_y_y)

    return num_cov_params + 2, predict, log_marginal_likelihood, unpack_kernel_params


class GP:
    def __init__(self, kernel):
        self.kernel = kernel
        self.rs = npr.RandomState(0)

        # Build model
        (
            num_params,
            predict,
            log_marginal_likelihood,
            unpack_kernel_params,
        ) = make_gp_funs(
            self.kernel, num_cov_params=2
        )  # params here are length scale, output scale

        self.num_params = num_params
        self.predict_fn = predict
        self.log_marginal_likelihood = log_marginal_likelihood

    def callback(self, params):
        pass

    def fit(self, X, y):

        self.X = X
        self.y = y

        objective = lambda params: -self.log_marginal_likelihood(params, self.X, self.y)

        # Initialize covariance parameters
        init_params = 0.1 * self.rs.randn(self.num_params)

        res = minimize(
            value_and_grad(objective),
            init_params,
            jac=True,
            method="CG",
            callback=self.callback,
        )

        self.params = res.x

    def predict(self, X_test):

        preds_mean, preds_cov = self.predict_fn(self.params, self.X, self.y, X_test)
        return preds_mean, preds_cov


class HGP:
    def __init__(self, within_group_kernel, between_group_kernel):
        self.kernel = lambda params, X1, X2, groups1, groups2: hgp_kernel(
            params, X1, X2, groups1, groups2, within_group_kernel, between_group_kernel
        )
        self.rs = npr.RandomState(0)

        # Build model
        (
            num_params,
            predict,
            log_marginal_likelihood,
            unpack_kernel_params,
        ) = make_gp_funs(
            self.kernel, num_cov_params=2 * 2, is_hgp=True
        )  # params here are length scale, output scale (for w/in group and b/w group)

        self.num_params = num_params
        self.predict_fn = predict
        self.log_marginal_likelihood = log_marginal_likelihood

    def callback(self, params):
        # print(params)
        pass

    def fit(self, X, y, groups):

        self.X = X
        self.y = y
        self.groups = groups

        objective = lambda params: -self.log_marginal_likelihood(
            params, self.X, self.y, groups=groups
        )

        # Initialize covariance parameters
        init_params = 0.1 * self.rs.randn(self.num_params)

        res = minimize(
            value_and_grad(objective),
            init_params,
            jac=True,
            method="CG",
            callback=self.callback,
        )

        self.params = res.x

    def predict(self, X_test, groups_test):

        preds_mean, preds_cov = self.predict_fn(
            self.params,
            self.X,
            self.y,
            X_test,
            x_groups=self.groups,
            xstar_groups=groups_test,
        )
        return preds_mean, preds_cov


class MGGP:
    def __init__(self, kernel, num_cov_params=3):
        self.kernel = kernel
        self.rs = npr.RandomState(0)

        # Build model
        (
            num_params,
            predict,
            log_marginal_likelihood,
            unpack_kernel_params,
        ) = make_gp_funs(
            self.kernel, num_cov_params=num_cov_params, is_mggp=True
        )  # params here are length scale, output scale, and a (group diff param)

        self.num_params = num_params
        self.predict_fn = predict
        self.log_marginal_likelihood = log_marginal_likelihood

    def callback(self, params):
        pass

    def fit(self, X, y, groups, group_distances=None):

        self.X = X
        self.y = y
        self.groups = groups
        self.n_groups = self.groups.shape[1]
        if group_distances is None:
            self.group_distances = np.ones((self.n_groups, self.n_groups))
        else:
            self.group_distances = group_distances

        objective = lambda params: -self.log_marginal_likelihood(
            params,
            self.X,
            self.y,
            groups=self.groups,
            group_distances=self.group_distances,
        )

        # Initialize covariance parameters
        init_params = 0.1 * self.rs.randn(self.num_params)

        lower_bound = np.log(1e-4)
        upper_bound = np.log(1e4)
        res = minimize(
            value_and_grad(objective),
            init_params,
            jac=True,
            # method="CG",
            callback=self.callback,
            bounds=[
                (None, None),
                (None, None),
                (lower_bound, upper_bound),
                (lower_bound, upper_bound),
                (lower_bound, upper_bound),
            ],
        )

        self.params = res.x

    def predict(self, X_test, groups_test):

        preds = self.predict_fn(
            self.params,
            self.X,
            self.y,
            X_test,
            x_groups=self.groups,
            xstar_groups=groups_test,
            group_distances=self.group_distances,
        )
        return preds


if __name__ == "__main__":

    import sys

    sys.path.append("../kernels")
    from kernels import multigroup_rbf_covariance

    n0, n1 = 50, 50
    p = 1
    n = n0 + n1
    noise_scale_true = 0.5
    X = np.random.uniform(low=-10, high=10, size=(n, p))

    X0_groups = np.repeat([[1, 0]], n0, axis=0)
    X1_groups = np.repeat([[0, 1]], n1, axis=0)
    X_groups = np.concatenate([X0_groups, X1_groups], axis=0)

    a = 1e0
    group_dist_mat = np.ones((2, 2))
    K_XX = multigroup_rbf_covariance(
        [np.log(1.0), np.log(a), np.log(1.0)],
        X,
        X,
        X_groups,
        X_groups,
        group_dist_mat,
    )
    Y = mvnpy.rvs(np.zeros(n), K_XX) + np.random.normal(scale=noise_scale_true, size=n)

    mggp = MGGP(kernel=multigroup_rbf_covariance)
    mggp.fit(X, Y, groups=X_groups)

    n_test = 100
    X_test = np.expand_dims(np.linspace(-10, 10, n_test), 1)
    preds_mean_g0, _ = mggp.predict(X_test, np.repeat([[1, 0]], n_test, axis=0))
    preds_mean_g1, _ = mggp.predict(X_test, np.repeat([[0, 1]], n_test, axis=0))

    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.scatter(
        X[X_groups[:, 0] == 1], Y[X_groups[:, 0] == 1], label="Group 1", color="blue"
    )
    plt.scatter(
        X[X_groups[:, 1] == 1], Y[X_groups[:, 1] == 1], label="Group 2", color="orange"
    )
    plt.title("Data")
    plt.legend()
    plt.subplot(212)
    plt.scatter(
        X[X_groups[:, 0] == 1], Y[X_groups[:, 0] == 1], label="Group 1", color="blue"
    )
    plt.scatter(
        X[X_groups[:, 1] == 1], Y[X_groups[:, 1] == 1], label="Group 2", color="orange"
    )
    plt.plot(X_test, preds_mean_g0, color="blue")
    plt.plot(X_test, preds_mean_g1, color="orange")
    plt.title("Predictions")
    plt.legend()
    plt.show()

    import ipdb

    ipdb.set_trace()
