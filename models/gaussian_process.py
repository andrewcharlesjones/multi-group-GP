from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import value_and_grad
from scipy.optimize import minimize

from scipy.stats import multivariate_normal as mvnpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.gaussian_process.kernels import RBF

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

    def predict(params, x, y, xstar, **kwargs):
        """Returns the predictive mean and covariance at locations xstar,
        of the latent function value f (without observation noise)."""
        mean, cov_params, noise_scale = unpack_kernel_params(params)

        if is_hgp:
            extra_args = {"groups1": kwargs['xstar_groups'], "groups2": kwargs['xstar_groups']}
        elif is_mggp:
            extra_args = {"groups1": kwargs['xstar_groups'], "groups2": kwargs['xstar_groups'], "group_distances": kwargs['group_distances']}
        else:
            extra_args = {}

        cov_f_f = cov_func(cov_params, xstar, xstar, **extra_args)

        if is_hgp:
            extra_args = {"groups1": kwargs['x_groups'], "groups2": kwargs['xstar_groups']}
        elif is_mggp:
            extra_args = {"groups1": kwargs['x_groups'], "groups2": kwargs['xstar_groups'], "group_distances": kwargs['group_distances']}

        cov_y_f = cov_func(cov_params, x, xstar, **extra_args)

        if is_hgp:
            extra_args = {"groups1": kwargs['x_groups'], "groups2": kwargs['x_groups']}
        elif is_mggp:
            extra_args = {"groups1": kwargs['x_groups'], "groups2": kwargs['x_groups'], "group_distances": kwargs['group_distances']}

        cov_y_y = cov_func(cov_params, x, x, **extra_args) + noise_scale * np.eye(len(y))

        pred_mean = mean + np.dot(solve(cov_y_y, cov_y_f).T, y - mean)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov

    def log_marginal_likelihood(params, x, y, **kwargs):

        mean, cov_params, noise_scale = unpack_kernel_params(params)
        if is_hgp:
            extra_args = {"groups1": kwargs['groups'], "groups2": kwargs['groups']}
        elif is_mggp:
            extra_args = {"groups1": kwargs['groups'], "groups2": kwargs['groups'], "group_distances": kwargs['group_distances']}
        else:
            extra_args = {}
        # import ipdb; ipdb.set_trace()
        cov_y_y = cov_func(cov_params, x, x, **extra_args) + noise_scale * np.eye(len(y))
        prior_mean = mean * np.ones(len(y))
        return mvn.logpdf(y, prior_mean, cov_y_y)

    return num_cov_params + 2, predict, log_marginal_likelihood, unpack_kernel_params


# Define an example covariance function.
def rbf_covariance(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])

    diffs = np.expand_dims(x / lengthscales, 1) - np.expand_dims(xp / lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))


def multigroup_rbf_covariance(kernel_params, X1, X2, groups1, groups2, group_distances):
    output_scale = np.exp(kernel_params[0])
    group_diff_param = np.exp(kernel_params[1])  # + 1e-4
    lengthscales = np.exp(kernel_params[2:])  # + 1e-6

    assert X1.shape[1] == X2.shape[1]
    assert groups1.shape[1] == groups2.shape[1]
    assert group_distances.shape[1] == groups1.shape[1]

    p = X1.shape[1]
    n_groups = groups1.shape[1]
    # import ipdb; ipdb.set_trace()

    diffs = np.expand_dims(X1 / lengthscales, 1) - np.expand_dims(X2 / lengthscales, 0)
    dists = np.sum(diffs ** 2, axis=2)

    diff_group_indicator = (
        np.expand_dims(groups1, 1) - np.expand_dims(groups2, 0)
    ) ** 2
    diff_group_scaling_term = np.zeros(dists.shape)

    for ii in range(n_groups):

        for jj in range(ii):

            if ii == jj:
                continue

            curr_group_distance = group_distances[ii, jj]
            

            diff_group_scaling_term += (curr_group_distance * group_diff_param ** 2 + 1) * (
                np.logical_and(
                    diff_group_indicator[:, :, ii] == 1, diff_group_indicator[:, :, jj] == 1
                )
            ).astype(int)

    samegroup_mask = (diff_group_scaling_term == 0).astype(int)
    diff_group_scaling_term += samegroup_mask

    dists /= diff_group_scaling_term

    K = output_scale * np.exp(-0.5 * dists)
    K /= (diff_group_scaling_term) ** (0.5 * p)

    return K


def matern12_covariance(kernel_params, x, xp):

    output_scale = np.exp(kernel_params[0])
    # group_diff_param = np.exp(kernel_params[1])# + 0.0001
    lengthscales = np.exp(kernel_params[1:])
    c = 1.0

    assert x.shape[1] == xp.shape[1]
    p = x.shape[1]

    # x_groups = x[:, -1]
    # x = x[:, :-1]
    # xp_groups = xp[:, -1]
    # xp = xp[:, :-1]
    #
    diffs = (
        np.expand_dims(x / lengthscales, 1)
        - np.expand_dims(xp / lengthscales, 0)
        + 1e-6
    )

    dists = np.linalg.norm(diffs, axis=2, ord=2)

    exponential_term = np.exp(-((1 / c) ** 0.5) * dists)
    premult_term = output_scale

    # K = output_scale * c**(0.5 * p) * (1 + np.sqrt(3) * dists) * np.exp(-np.sqrt(3) * dists)
    # K /= ((diff_group_scaling_term)**(0.5 * p))

    K = premult_term * exponential_term
    # import ipdb ;ipdb.set_trace()

    return K


def mg_matern12_covariance(kernel_params, x, xp):

    output_scale = np.exp(kernel_params[0])
    group_diff_param = np.exp(kernel_params[1])  # + 0.0001
    lengthscales = np.exp(kernel_params[2:])
    c = 1.0

    assert x.shape[1] == xp.shape[1]
    p = x.shape[1] - 1

    x_groups = x[:, -1]
    x = x[:, :-1]
    xp_groups = xp[:, -1]
    xp = xp[:, :-1]

    diffs = (
        np.expand_dims(x / lengthscales, 1)
        - np.expand_dims(xp / lengthscales, 0)
        + 1e-6
    )
    # import ipdb ;ipdb.set_trace()
    # dists = np.sqrt(np.sum(diffs**2, axis=2))
    dists = np.linalg.norm(diffs, axis=2, ord=2)

    diff_group_indicator = (
        np.expand_dims(x_groups, 1) - np.expand_dims(xp_groups, 0)
    ) ** 2
    diff_group_scaling_term = diff_group_indicator * group_diff_param ** 2
    # dists /= diff_group_scaling_term

    exponential_term = np.exp(
        -(diff_group_scaling_term + 1)
        / (output_scale * c ** (0.5 * p) + c) ** 0.5
        * dists
    )
    premult_term = (
        output_scale
        * c ** (0.5 * p)
        / (
            (diff_group_scaling_term + 1) ** 0.5
            * (diff_group_scaling_term + c) ** (0.5 * c)
        )
    )

    # K = output_scale * c**(0.5 * p) * (1 + np.sqrt(3) * dists) * np.exp(-np.sqrt(3) * dists)
    # K /= ((diff_group_scaling_term)**(0.5 * p))

    K = premult_term * exponential_term
    # import ipdb ;ipdb.set_trace()

    return K


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


def hgp_kernel(params, X1, X2, groups1, groups2, within_group_kernel, between_group_kernel):

    diff_group_indicator = (
        np.expand_dims(groups1, 1) - np.expand_dims(groups2, 0)
    ) ** 2
    same_group_mask = (np.sum(diff_group_indicator, axis=2) == 0).astype(int)    

    n_params_total = len(params)
    within_group_params = params[:n_params_total//2]
    between_group_params = params[n_params_total//2:]

    K_within = within_group_kernel(within_group_params, X1, X2)
    K_between = between_group_kernel(between_group_params, X1, X2)

    K = K_between + same_group_mask * K_within

    return K


class HGP:
    def __init__(self, within_group_kernel, between_group_kernel):
        self.kernel = lambda params, X1, X2, groups1, groups2: hgp_kernel(params, X1, X2, groups1, groups2, within_group_kernel, between_group_kernel)
        self.rs = npr.RandomState(0)

        # Build model
        (
            num_params,
            predict,
            log_marginal_likelihood,
            unpack_kernel_params,
        ) = make_gp_funs(
            self.kernel, num_cov_params=2*2, is_hgp=True
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

        objective = lambda params: -self.log_marginal_likelihood(params, self.X, self.y, groups=groups)

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

        preds_mean, preds_cov = self.predict_fn(self.params, self.X, self.y, X_test, x_groups=self.groups, xstar_groups=groups_test)
        return preds_mean, preds_cov


class MGGP:
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
            self.kernel, num_cov_params=3, is_mggp=True
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



        objective = lambda params: -self.log_marginal_likelihood(params, self.X, self.y, groups=self.groups, group_distances=self.group_distances)

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

        preds_mean, preds_cov = self.predict_fn(self.params, self.X, self.y, X_test, x_groups=self.groups, xstar_groups=groups_test, group_distances=self.group_distances)
        return preds_mean, preds_cov


if __name__ == "__main__":

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
    Y = mvnpy.rvs(np.zeros(n), K_XX) + np.random.normal(
        scale=noise_scale_true, size=n
    )

    mggp = MGGP(kernel=multigroup_rbf_covariance)
    mggp.fit(X, Y, groups=X_groups)

    n_test = 100
    X_test = np.expand_dims(np.linspace(-10, 10, n_test), 1)
    preds_mean_g0, _ = mggp.predict(X_test, np.repeat([[1, 0]], n_test, axis=0))
    preds_mean_g1, _ = mggp.predict(X_test, np.repeat([[0, 1]], n_test, axis=0))


    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.scatter(X[X_groups[:, 0] == 1], Y[X_groups[:, 0] == 1], label="Group 1", color="blue")
    plt.scatter(X[X_groups[:, 1] == 1], Y[X_groups[:, 1] == 1], label="Group 2", color="orange")
    plt.title("Data")
    plt.legend()
    plt.subplot(212)
    plt.scatter(X[X_groups[:, 0] == 1], Y[X_groups[:, 0] == 1], label="Group 1", color="blue")
    plt.scatter(X[X_groups[:, 1] == 1], Y[X_groups[:, 1] == 1], label="Group 2", color="orange")
    plt.plot(X_test, preds_mean_g0, color="blue")
    plt.plot(X_test, preds_mean_g1, color="orange")
    plt.title("Predictions")
    plt.legend()
    plt.show()


    import ipdb

    ipdb.set_trace()
