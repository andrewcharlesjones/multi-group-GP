import autograd.numpy as np
import autograd.numpy.random as npr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
import autograd.scipy.stats.multivariate_normal as mvn
from scipy.stats import multivariate_normal as mvnpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from autograd import value_and_grad
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm


sys.path.append("../models")
from gaussian_process import make_gp_funs, mg_rbf_covariance, rbf_covariance

sys.path.append("../kernels")
from mgRBF import mgRBF

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


n_repeats = 10
p = 1
noise_scale_true = 0.1
output_scale_true = 1.0
length_scale_true = 1.0
alpha_true = 0.1
n0 = 100
n1 = 100
alpha_list = alpha_list = [np.power(10, x * 1.0) for x in np.arange(-3, 5)]
# FIX_TRUE_PARAMS = False
FIX_ALPHA = True


lengthscales_mggp = np.empty((n_repeats, len(alpha_list)))
lengthscales_union_gp = np.empty((n_repeats, len(alpha_list)))

def separated_gp():
	
	fitted_as = np.empty((n_repeats, len(alpha_list)))

	for ii in tqdm(range(n_repeats)):

		for jj, curr_alpha in enumerate(alpha_list):
	
			## Generate data
			X0 = np.random.uniform(-10, 10, size=(n0, p))
			X1 = np.random.uniform(-10, 10, size=(n1, p))
			X0_groups = np.zeros((n0, 1))
			X1_groups = np.ones((n1, 1))
			X0 = np.hstack([X0, X0_groups])
			X1 = np.hstack([X1, X1_groups])
			X = np.concatenate([X0, X1], axis=0)
			K_XX = mg_rbf_covariance([1., np.log(alpha_true), length_scale_true], X, X)
			Y = mvnpy.rvs(np.zeros(n0 + n1), K_XX) + np.random.normal(scale=noise_scale_true, size=n0 + n1)

			# Build model and objective function.
			num_params, predict, log_marginal_likelihood, unpack_kernel_params = \
				make_gp_funs(mg_rbf_covariance, num_cov_params=2 if FIX_ALPHA else 3) # params here are length scale, output scale, and a (group diff param)

			if FIX_ALPHA:
				objective = lambda params: -log_marginal_likelihood(np.concatenate([params[:3], [np.log(curr_alpha - 0.0001)], params[3:]]), X, Y)
			else:
				objective = lambda params: -log_marginal_likelihood(params, X, Y)

			# Initialize covariance parameters
			rs = npr.RandomState(0)

			init_params = 0.1 * rs.randn(num_params)

			def callback(params):
				pass

			res_mggp = minimize(value_and_grad(objective), init_params, jac=True,
								  method='CG', callback=callback)

			lengthscales_mggp[ii, jj] = res_mggp.x[-1]

			## Fit union GP
			# objective = lambda params: -log_marginal_likelihood(params, X[:, :-1], Y)

			# num_params, predict, log_marginal_likelihood, unpack_kernel_params = \
			# 			make_gp_funs(rbf_covariance, num_cov_params=2) # params here are length scale and output scale

			# rs = npr.RandomState(0)
			# init_params = 0.1 * rs.randn(num_params)

			# res_union_gp = minimize(value_and_grad(objective), init_params, jac=True,
			# 					  method='CG', callback=callback)
			# lengthscales_union_gp[ii, jj] = res_union_gp.x[-1]



	results_df_mggp = pd.melt(pd.DataFrame(lengthscales_mggp, columns=alpha_list))
	results_df_mggp["method"] = ["MGGP"] * results_df_mggp.shape[0]

	# results_df_union_gp = pd.melt(pd.DataFrame(lengthscales_union_gp, columns=alpha_list))
	# results_df_union_gp["method"] = ["Union GP"] * results_df_union_gp.shape[0]

	# results_df = pd.concat([results_df_mggp, results_df_union_gp], axis=0)

	plt.figure(figsize=(7, 5))
	sns.boxplot(data=results_df_mggp, x="variable", y="value") #, hue="method")
	plt.xlabel(r"Model's $a$")
	plt.ylabel(r"Estimated $b$")
	plt.title(r"True $a$: $" + str(alpha_true) + "$")
	plt.tight_layout()
	plt.savefig("../plots/lengthscale_recovery.png")
	plt.show()

	import ipdb; ipdb.set_trace()

	return fitted_as


if __name__ == '__main__':

	# plt.figure(figsize=(14, 5))


	# plt.subplot(121)
	fitted_as_sep = np.exp(separated_gp())
	# fitted_as_union = np.sqrt(np.exp(union_gp()))

	# results_df = pd.melt(pd.DataFrame({"Separated GP": fitted_as_sep, "Union GP": fitted_as_union}))

	
	# sns.boxplot(data=results_df, x="variable", y="value")
	# plt.xlabel("Model from which data is generated")
	# plt.ylabel(r"Fitted $a$")
	# plt.yscale("log")
	# plt.title(r"Only $a$ estimated")
	# plt.tight_layout()

	# plt.subplot(122)
	# FIX_TRUE_PARAMS = False
	# fitted_as_sep = np.exp(separated_gp())
	# fitted_as_union = np.exp(union_gp())

	# results_df = pd.melt(pd.DataFrame({"Separated GP": fitted_as_sep, "Union GP": fitted_as_union}))

	
	# sns.boxplot(data=results_df, x="variable", y="value")
	# plt.xlabel("Model from which data is generated")
	# plt.ylabel(r"Fitted $a$")
	# plt.yscale("log")
	# plt.title("All parameters estimated")
	# plt.tight_layout()
	

	# plt.savefig("../plots/alpha_optimized.png")
	# # if FIX_TRUE_PARAMS:
	# # 	plt.savefig("../plots/alpha_optimized_fixed_true_params.png")
	# # else:
	# # 	plt.savefig("../plots/alpha_optimized.png")
	# plt.show()

	# plt.hist(fitted_as_sep, label="Separated GP")
	# plt.hist(fitted_as_union, label="Union GP")
	# plt.legend()
	# plt.show()
	import ipdb; ipdb.set_trace()
	

