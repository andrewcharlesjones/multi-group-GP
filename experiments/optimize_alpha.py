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

sys.path.append("../models")
from gaussian_process import make_gp_funs, mg_rbf_covariance

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


n_repeats = 10
p = 1
noise_scale_true = 0.1
n0 = 200
n1 = 200
# alpha_list = alpha_list = [np.power(10, x * 1.0) for x in np.arange(-5, 5)]
FIX_TRUE_PARAMS = False


def separated_gp():
	
	fitted_as = np.empty(n_repeats)

	for ii in range(n_repeats):
	
		## Generate data
		X0 = np.random.uniform(low=-10, high=10, size=(n0, p))
		X1 = np.random.uniform(low=-10, high=10, size=(n1, p))
		kernel = RBF()
		K_X0X0 = kernel(X0, X0)
		K_X1X1 = kernel(X1, X1)
		Y0 = mvnpy.rvs(np.zeros(n0), K_X0X0) + np.random.normal(scale=noise_scale_true, size=n0)
		Y1 = mvnpy.rvs(np.zeros(n1), K_X1X1) + np.random.normal(scale=noise_scale_true, size=n1)

		X0_groups = np.zeros((n0, 1))
		X1_groups = np.ones((n1, 1))
		X0 = np.hstack([X0, X0_groups])
		X1 = np.hstack([X1, X1_groups])
		X = np.concatenate([X0, X1], axis=0)
		Y = np.concatenate([Y0, Y1])

		# Build model and objective function.
		num_params, predict, log_marginal_likelihood, unpack_kernel_params = \
			make_gp_funs(mg_rbf_covariance, num_cov_params=3) # params here are length scale, output scale, and a (group diff param)

		if FIX_TRUE_PARAMS:
			objective = lambda a: -log_marginal_likelihood(np.concatenate([[0., np.log(noise_scale_true - 0.0001), np.log(1.)], a, [np.log(1.)]]), X, Y)
		else:
			objective = lambda params: -log_marginal_likelihood(params, X, Y)

		# Initialize covariance parameters
		rs = npr.RandomState(0)

		if FIX_TRUE_PARAMS:
			init_params = 1.
		else:
			init_params = 0.1 * rs.randn(num_params)

		def callback(params):
			pass

		res = minimize(value_and_grad(objective), init_params, jac=True,
							  method='CG', callback=callback)

		if FIX_TRUE_PARAMS:
			curr_a = res.x
		else:
			mean, cov_params, noise_scale = unpack_kernel_params(res.x)
			output_scale = np.exp(cov_params[0])
			group_diff_param = np.exp(cov_params[1])
			lengthscales = np.exp(cov_params[2:])
			curr_a = np.log(group_diff_param)

		fitted_as[ii] = curr_a

	return fitted_as

def union_gp():

	fitted_as = np.empty(n_repeats)

	for ii in range(n_repeats):
	
		## Generate data
		X = np.random.uniform(low=-10, high=10, size=(n0 + n1, p))
		kernel = RBF()
		K_XX = kernel(X, X)
		Y = mvnpy.rvs(np.zeros(n0 + n1), K_XX) + np.random.normal(scale=noise_scale_true, size=n0 + n1)

		groups = np.concatenate([np.zeros((n0, 1)), np.ones((n1, 1))])
		X = np.hstack([X, groups])

		# Build model and objective function.
		num_params, predict, log_marginal_likelihood, unpack_kernel_params = \
			make_gp_funs(mg_rbf_covariance, num_cov_params=3) # params here are length scale, output scale, and a (group diff param)


		if FIX_TRUE_PARAMS:
			objective = lambda a: -log_marginal_likelihood(np.concatenate([[0., np.log(noise_scale_true - 0.0001), np.log(1.)], a, [np.log(1.)]]), X, Y)
		else:
			objective = lambda params: -log_marginal_likelihood(params, X, Y)

		# Initialize covariance parameters
		rs = npr.RandomState(0)

		if FIX_TRUE_PARAMS:
			init_params = 1.
		else:
			init_params = 0.1 * rs.randn(num_params)

		def callback(params):
			pass

		res = minimize(value_and_grad(objective), init_params, jac=True,
							  method='CG', callback=callback)

		if FIX_TRUE_PARAMS:
			curr_a = res.x
		else:
			mean, cov_params, noise_scale = unpack_kernel_params(res.x)
			output_scale = np.exp(cov_params[0])
			group_diff_param = np.exp(cov_params[1])
			lengthscales = np.exp(cov_params[2:])
			curr_a = np.log(group_diff_param)
			
		fitted_as[ii] = curr_a

	return fitted_as

if __name__ == '__main__':

	plt.figure(figsize=(14, 5))


	plt.subplot(121)
	FIX_TRUE_PARAMS = True
	fitted_as_sep = np.sqrt(np.exp(separated_gp()))
	fitted_as_union = np.sqrt(np.exp(union_gp()))

	results_df = pd.melt(pd.DataFrame({"Separated GP": fitted_as_sep, "Union GP": fitted_as_union}))

	
	sns.boxplot(data=results_df, x="variable", y="value")
	plt.xlabel("Model from which data is generated")
	plt.ylabel(r"Fitted $a$")
	plt.yscale("log")
	plt.title(r"Only $a$ estimated")
	plt.tight_layout()

	plt.subplot(122)
	FIX_TRUE_PARAMS = False
	fitted_as_sep = np.exp(separated_gp())
	fitted_as_union = np.exp(union_gp())

	results_df = pd.melt(pd.DataFrame({"Separated GP": fitted_as_sep, "Union GP": fitted_as_union}))

	
	sns.boxplot(data=results_df, x="variable", y="value")
	plt.xlabel("Model from which data is generated")
	plt.ylabel(r"Fitted $a$")
	plt.yscale("log")
	plt.title("All parameters estimated")
	plt.tight_layout()
	

	plt.savefig("../plots/alpha_optimized.png")
	# if FIX_TRUE_PARAMS:
	# 	plt.savefig("../plots/alpha_optimized_fixed_true_params.png")
	# else:
	# 	plt.savefig("../plots/alpha_optimized.png")
	plt.show()

	# plt.hist(fitted_as_sep, label="Separated GP")
	# plt.hist(fitted_as_union, label="Union GP")
	# plt.legend()
	# plt.show()
	import ipdb; ipdb.set_trace()
	

