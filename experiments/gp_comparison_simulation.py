import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append("../kernels")
from mgRBF import mgRBF
from RBF import RBF
from RBF_groupwise import RBF_groupwise

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

n_repeats = 20
alpha_list = alpha_list = [np.power(10, x * 1.0) for x in np.arange(-15, 1)]

def separate_GP_experiment():
	## Generate data from independent GP for each group

	n0 = 100
	n1 = 90
	p = 1
	
	ll_mggp_results = np.empty((n_repeats, len(alpha_list)))
	ll_sep_gp_results = np.empty(n_repeats)
	ll_union_gp_results = np.empty(n_repeats)
	for ii in range(n_repeats):

		## Generate data
		X0 = np.random.normal(size=(n0, p))
		X1 = np.random.normal(size=(n1, p))
		kernel = RBF()
		K_X0X0 = kernel(X0, X0)
		K_X1X1 = kernel(X1, X1)
		Y0 = mvn.rvs(np.zeros(n0), K_X0X0)
		Y1 = mvn.rvs(np.zeros(n1), K_X1X1)

		## Fit independent GPs
		gpr = GaussianProcessRegressor(kernel=RBF(), optimizer=None)
		ll0 = gpr.fit(X0, Y0).log_marginal_likelihood()
		ll1 = gpr.fit(X1, Y1).log_marginal_likelihood()
		ll_independent = ll0 + ll1
		ll_sep_gp_results[ii] = ll_independent

		## Fit union GP
		X = np.concatenate([X0, X1], axis=0)
		Y = np.concatenate([Y0, Y1])
		gpr = GaussianProcessRegressor(kernel=RBF(), optimizer=None)
		ll = gpr.fit(X, Y).log_marginal_likelihood()
		ll_union_gp_results[ii] = ll

		for jj, alpha in enumerate(alpha_list):

			## Fit MGGP
			X0_groups = np.zeros((n0, 1))
			X1_groups = np.ones((n1, 1))
			X0 = np.hstack([X0, X0_groups])
			X1 = np.hstack([X1, X1_groups])
			X = np.concatenate([X0, X1], axis=0)
			Y = np.concatenate([Y0, Y1])
			gpr = GaussianProcessRegressor(kernel=mgRBF(group_diff_param=alpha), optimizer=None)
			ll_mggp = gpr.fit(X, Y).log_marginal_likelihood()
			ll_mggp_results[ii, jj] = ll_mggp

	## Plot MGGP results
	results_df = pd.melt(pd.DataFrame(ll_mggp_results, columns=alpha_list))
	sns.lineplot(data=results_df, x="variable", y="value", label="MGGP", ci="sd")

	## Plot sep GP results
	gp_results_df = pd.melt(pd.DataFrame(np.vstack([ll_sep_gp_results, ll_sep_gp_results]).T, columns=[alpha_list[0], alpha_list[-1]]))
	sns.lineplot(data=gp_results_df, x="variable", y="value", label="Separate GPs", color="red", ci="sd")

	## Plot union GP results
	gp_results_df = pd.melt(pd.DataFrame(np.vstack([ll_union_gp_results, ll_union_gp_results]).T, columns=[alpha_list[0], alpha_list[-1]]))
	sns.lineplot(data=gp_results_df, x="variable", y="value", label="Union GPs", color="green", ci="sd")

	plt.xscale("log")
	plt.xlabel(r"$\alpha^2$")
	plt.ylabel(r"$\log p(Y)$")
	plt.legend(loc="lower right")
	plt.tight_layout()
	# plt.savefig("../plots/separated_gp_comparison.png")
	# plt.show()

	# import ipdb; ipdb.set_trace()

def union_GP_experiment():
	## Generate data from independent GP for each group

	n0 = 100
	n1 = 90
	p = 1
	
	ll_mggp_results = np.empty((n_repeats, len(alpha_list)))
	ll_sep_gp_results = np.empty(n_repeats)
	ll_union_gp_results = np.empty(n_repeats)
	for ii in range(n_repeats):

		## Generate data
		X0 = np.random.normal(size=(n0, p))
		X1 = np.random.normal(size=(n1, p))
		X = np.concatenate([X0, X1], axis=0)
		kernel = RBF()
		K_XX = kernel(X, X)
		Y = mvn.rvs(np.zeros(n0 + n1), K_XX)
		Y0, Y1 = Y[:n0], Y[n0:]
		
		Y = np.concatenate([Y0, Y1])

		## Fit independent GPs
		gpr = GaussianProcessRegressor(kernel=RBF(), optimizer=None)
		ll0 = gpr.fit(X0, Y0).log_marginal_likelihood()
		ll1 = gpr.fit(X1, Y1).log_marginal_likelihood()
		ll_independent = ll0 + ll1
		ll_sep_gp_results[ii] = ll_independent

		## Fit union GP
		gpr = GaussianProcessRegressor(kernel=RBF(), optimizer=None)
		ll = gpr.fit(X, Y).log_marginal_likelihood()
		ll_union_gp_results[ii] = ll

		for jj, alpha in enumerate(alpha_list):

			## Fit MGGP
			X0_groups = np.zeros((n0, 1))
			X1_groups = np.ones((n1, 1))
			X0 = np.hstack([X0, X0_groups])
			X1 = np.hstack([X1, X1_groups])
			X = np.concatenate([X0, X1], axis=0)
			Y = np.concatenate([Y0, Y1])
			gpr = GaussianProcessRegressor(kernel=mgRBF(group_diff_param=alpha), optimizer=None)
			ll_mggp = gpr.fit(X, Y).log_marginal_likelihood()
			ll_mggp_results[ii, jj] = ll_mggp

	## Plot MGGP results
	results_df = pd.melt(pd.DataFrame(ll_mggp_results, columns=alpha_list))
	sns.lineplot(data=results_df, x="variable", y="value", label="MGGP", ci="sd")

	## Plot sep GP results
	gp_results_df = pd.melt(pd.DataFrame(np.vstack([ll_sep_gp_results, ll_sep_gp_results]).T, columns=[alpha_list[0], alpha_list[-1]]))
	sns.lineplot(data=gp_results_df, x="variable", y="value", label="Separate GPs", color="red", ci="sd")

	## Plot union GP results
	gp_results_df = pd.melt(pd.DataFrame(np.vstack([ll_union_gp_results, ll_union_gp_results]).T, columns=[alpha_list[0], alpha_list[-1]]))
	sns.lineplot(data=gp_results_df, x="variable", y="value", label="Union GPs", color="green", ci="sd")

	plt.xscale("log")
	plt.xlabel(r"$\alpha^2$")
	plt.ylabel(r"$\log p(Y)$")
	plt.legend(loc="lower right")
	plt.tight_layout()
	# plt.savefig("../plots/union_gp_comparison.png")
	# plt.show()

	# import ipdb; ipdb.set_trace()

def HGP_experiment():
	## Generate data from independent GP for each group

	n0 = 100
	n1 = 90
	p = 1

	length_scale_shared = 1.
	length_scale_group0 = 10.
	length_scale_group1 = 0.1
	
	alpha_list = [np.power(10, x * 1.0) for x in np.arange(-5, 5)]
	n_repeats = 20
	ll_mggp_results = np.empty((n_repeats, len(alpha_list)))
	ll_gp_results = np.empty(n_repeats)
	for ii in range(n_repeats):

		## Generate data
		X0 = np.random.normal(scale=1., size=(n0, p))
		X1 = np.random.normal(scale=1., size=(n1, p))
		X = np.concatenate([X0, X1], axis=0)
		kernel_shared = RBF(length_scale=1.0)
		kernel_group0 = RBF(length_scale=length_scale_group0)
		kernel_group1 = RBF(length_scale=length_scale_group1)

		K_XX = kernel_shared(X, X)
		K_XX[:n0, :n0] += kernel_group0(X0, X0)
		K_XX[n0:, n0:] += kernel_group1(X1, X1)
		Y = mvn.rvs(np.zeros(n0 + n1), K_XX)

		## Fit hierarchical GP
		X0_groups = np.zeros((n0, 1))
		X1_groups = np.ones((n1, 1))
		X0 = np.hstack([X0, X0_groups])
		X1 = np.hstack([X1, X1_groups])
		X = np.concatenate([X0, X1], axis=0)
		kernel_shared = RBF(length_scale=length_scale_shared, has_group=True)
		kernel_group_specific = RBF_groupwise(length_scale_group0=length_scale_group0, length_scale_group1=length_scale_group1)
		gpr = GaussianProcessRegressor(kernel=Sum(kernel_shared, kernel_group_specific), optimizer=None)
		ll = gpr.fit(X, Y).log_marginal_likelihood()
		ll_gp_results[ii] = ll

		for jj, alpha in enumerate(alpha_list):

			## Fit MGGP
			kernel_shared = mgRBF(length_scale=length_scale_shared, group_diff_param=alpha)
			kernel_group_specific = RBF_groupwise(length_scale_group0=length_scale_group0, length_scale_group1=length_scale_group1)
			curr_kernel = Sum(kernel_shared, kernel_group_specific)
			gpr = GaussianProcessRegressor(kernel=curr_kernel, optimizer=None)
			ll_mggp = gpr.fit(X, Y).log_marginal_likelihood()
			ll_mggp_results[ii, jj] = ll_mggp

	# plt.figure(figsize=(7, 5))
	results_df = pd.melt(pd.DataFrame(ll_mggp_results, columns=alpha_list))
	sns.lineplot(data=results_df, x="variable", y="value", label="MGGP", ci="sd")
	
	gp_results_df = pd.melt(pd.DataFrame(np.vstack([ll_gp_results, ll_gp_results]).T, columns=[alpha_list[0], alpha_list[-1]]))
	sns.lineplot(data=gp_results_df, x="variable", y="value", label="HGP", color="orange", ci="sd")
	plt.xscale("log")
	plt.xlabel(r"$\alpha^2$")
	plt.ylabel(r"$\log p(Y)$")
	plt.legend(loc="lower right")
	plt.tight_layout()
	# plt.savefig("../plots/hgp_comparison.png")
	# plt.show()

	# import ipdb; ipdb.set_trace()


def MGGP_experiment():
	## Generate data from independent GP for each group

	n0 = 100
	n1 = 90
	p = 1

	length_scale = 1.0
	alpha_true = 1e-8
	
	ll_mggp_results = np.empty((n_repeats, len(alpha_list)))
	ll_sep_gp_results = np.empty(n_repeats)
	ll_union_gp_results = np.empty(n_repeats)
	ll_hgp_results = np.empty(n_repeats)
	for ii in range(n_repeats):

		## Generate data
		X0 = np.random.normal(scale=1., size=(n0, p))
		X1 = np.random.normal(scale=1., size=(n1, p))
		X0_groups = np.zeros((n0, 1))
		X1_groups = np.ones((n1, 1))
		X0 = np.hstack([X0, X0_groups])
		X1 = np.hstack([X1, X1_groups])
		X = np.concatenate([X0, X1], axis=0)
		kernel = mgRBF(length_scale=length_scale, group_diff_param=alpha_true)
		gpr = GaussianProcessRegressor(kernel=kernel)
		Y = gpr.sample_y(X, n_samples=1)

		## Fit separate GPs
		gpr = GaussianProcessRegressor(kernel=RBF(length_scale=length_scale), optimizer=None)
		ll0 = gpr.fit(X[:n0, :1], Y[:n0]).log_marginal_likelihood()
		ll1 = gpr.fit(X[n0:, :1], Y[n0:]).log_marginal_likelihood()
		ll_sep_gp_results[ii] = ll0 + ll1

		## Fit union GP
		gpr = GaussianProcessRegressor(kernel=RBF(length_scale=length_scale), optimizer=None)
		ll = gpr.fit(X[:, :1], Y).log_marginal_likelihood()
		ll_union_gp_results[ii] = ll

		## Fit hierarchical GP
		# kernel_shared = RBF(length_scale=length_scale, has_group=True)
		# kernel_group_specific = RBF_groupwise(length_scale_group0=length_scale, length_scale_group1=length_scale)
		# gpr = GaussianProcessRegressor(kernel=Sum(kernel_shared, kernel_group_specific), optimizer=None)
		# ll = gpr.fit(X, Y).log_marginal_likelihood()
		# ll_hgp_results[ii] = ll

		for jj, alpha in enumerate(alpha_list):

			## Fit MGGP
			gpr = GaussianProcessRegressor(kernel=mgRBF(length_scale=length_scale, group_diff_param=alpha), optimizer=None)
			ll_mggp = gpr.fit(X, Y).log_marginal_likelihood()
			ll_mggp_results[ii, jj] = ll_mggp

	# plt.figure(figsize=(7, 5))
	results_df = pd.melt(pd.DataFrame(ll_mggp_results, columns=alpha_list))
	sns.lineplot(data=results_df, x="variable", y="value", label="MGGP", ci="sd")
	
	gp_results_df = pd.melt(pd.DataFrame(np.vstack([ll_sep_gp_results, ll_sep_gp_results]).T, columns=[alpha_list[0], alpha_list[-1]]))
	sns.lineplot(data=gp_results_df, x="variable", y="value", label="Separate GPs", color="red", ci="sd")

	gp_results_df = pd.melt(pd.DataFrame(np.vstack([ll_union_gp_results, ll_union_gp_results]).T, columns=[alpha_list[0], alpha_list[-1]]))
	sns.lineplot(data=gp_results_df, x="variable", y="value", label="Union GP", color="green", ci="sd")

	# gp_results_df = pd.melt(pd.DataFrame(np.vstack([ll_hgp_results, ll_hgp_results]).T, columns=[alpha_list[0], alpha_list[-1]]))
	# sns.lineplot(data=gp_results_df, x="variable", y="value", label="HGP", color="orange", ci="sd")

	plt.axvline(alpha_true, linestyle="--", alpha=0.3, color="black")

	# import ipdb; ipdb.set_trace()

	plt.xscale("log")
	# plt.yscale("log")
	plt.xlabel(r"$\alpha^2$")
	plt.ylabel(r"$\log p(Y)$")
	plt.legend(loc="lower right")
	plt.tight_layout()
	# plt.savefig("../plots/mggp_comparison.png")
	# plt.show()

	# import ipdb; ipdb.set_trace()

if __name__ == "__main__":
	plt.figure(figsize=(21, 7))
	plt.subplot(131)
	separate_GP_experiment()
	plt.title(r"Data generated from: $\textbf{Separated GP}$")
	# plt.show()
	# import ipdb; ipdb.set_trace()
	plt.subplot(132)
	union_GP_experiment()
	plt.title(r"Data generated from: $\textbf{Union GP}$")
	# plt.subplot(143)
	# HGP_experiment()
	# plt.title(r"Data generated from: $\textbf{Hierarchical GP}$")
	plt.subplot(133)
	MGGP_experiment()
	plt.title(r"Data generated from: $\textbf{Multi-group GP}$")
	plt.savefig("../plots/simulation_gp_comparison.png")
	plt.show()

	# MGGP_experiment()
	

