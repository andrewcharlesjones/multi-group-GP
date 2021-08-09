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



def MGGP_experiment():
	## Generate data from independent GP for each group

	n0 = 100
	n1 = 90
	p = 1

	length_scale = 1.0
	alpha_true = 1e-8
	
	alpha_list = [np.power(10, x * 1.0) for x in np.arange(-12, 5)]
	n_repeats = 20
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


		## Fit MGGP
		kernel = mgRBF(length_scale=10., group_diff_param=1.)
		# kernel = RBF(length_scale=10.)
		gpr = GaussianProcessRegressor(kernel=kernel)
		gpr.fit(X, Y)
		ll_mggp = gpr.log_marginal_likelihood()
		print(gpr.kernel_)
		# ll_mggp_results[ii, jj] = ll_mggp
		import ipdb; ipdb.set_trace()


	# plt.figure(figsize=(7, 5))
	results_df = pd.melt(pd.DataFrame(ll_mggp_results, columns=alpha_list))
	sns.lineplot(data=results_df, x="variable", y="value", label="MGGP", ci="sd")

	plt.axvline(alpha_true, linestyle="--", alpha=0.3, color="black")

	plt.xscale("log")
	# plt.yscale("log")
	plt.xlabel(r"$\alpha^2$")
	plt.ylabel(r"$\log p(Y)$")
	plt.legend(loc="lower right")
	plt.tight_layout()
	# plt.savefig("../plots/mggp_comparison.png")
	plt.show()

	# import ipdb; ipdb.set_trace()

if __name__ == "__main__":

	MGGP_experiment()
	

