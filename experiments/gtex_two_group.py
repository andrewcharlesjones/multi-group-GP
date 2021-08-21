# import numpy as np
import autograd.numpy as np
import autograd.numpy.random as npr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
from scipy.optimize import minimize
from autograd import value_and_grad

import sys
sys.path.append("../models")
from gaussian_process import make_gp_funs, mg_rbf_covariance, rbf_covariance
sys.path.append("../kernels")
from mgRBF import mgRBF
from mgMatern import mgMatern
from RBF import RBF
from RBF_groupwise import RBF_groupwise

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

DATA_DIR = "../data/gtex"
# DATA_FILE1 = "tibial_artery_expression.csv"
DATA_FILE1 = "frontal_cortex_expression.csv"
# DATA_FILE2 = "breast_mammary_expression.csv"
# OUTPUT_FILE1 = "tibial_artery_ischemic_time.csv"
OUTPUT_FILE1 = "frontal_cortex_ischemic_time.csv"
# OUTPUT_FILE2 = "breast_mammary_ischemic_time.csv"

# group1_tissue = "Tibial artery"
group1_tissue = "Anterior cingulate cortex"

data2_files = ["breast_mammary_expression.csv", "anterior_cingulate_cortex_expression.csv"]
output2_files = ["breast_mammary_ischemic_time.csv", "anterior_cingulate_cortex_ischemic_time.csv"]
# tissue_labels = ["Group 1: " + group1_tissue + "\nGroup 2: Breast", "Group 1: " + group1_tissue + "\nGroup 2: Coronary artery"]
tissue_labels = ["Group 1: " + group1_tissue + "\nGroup 2: Breast", "Group 1: " + group1_tissue + "\nGroup 2: Frontal cortex"]

# plt.figure(figsize=(7 * len(data2_files), 5))

n_repeats = 5
n_genes = 20
n_samples = 100
results = np.empty((n_repeats, len(tissue_labels)))

for ii in range(len(data2_files)):

	DATA_FILE2 = data2_files[ii]
	OUTPUT_FILE2 = output2_files[ii]

	## Load data
	data1 = pd.read_csv(pjoin(DATA_DIR, DATA_FILE1), index_col=0)
	data2 = pd.read_csv(pjoin(DATA_DIR, DATA_FILE2), index_col=0)

	output1 = pd.read_csv(pjoin(DATA_DIR, OUTPUT_FILE1), index_col=0)
	output2 = pd.read_csv(pjoin(DATA_DIR, OUTPUT_FILE2), index_col=0)

	assert np.array_equal(data1.index.values, output1.index.values)
	assert np.array_equal(data2.index.values, output2.index.values)

	X0, Y0 = data1.values, output1.values
	X1, Y1 = data2.values, output2.values


	## Log-transform and standardize data
	X0 = np.log(X0 + 1)
	X0 = (X0 - X0.mean(0)) / X0.std(0)
	Y0 = (Y0 - Y0.mean(0)) / Y0.std(0)

	X1 = np.log(X1 + 1)
	X1 = (X1 - X1.mean(0)) / X1.std(0)
	Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)

	n0, n1 = X0.shape[0], X1.shape[0]

	X0_groups = np.zeros((n0, 1))
	X1_groups = np.ones((n1, 1))
	X0 = np.hstack([X0, X0_groups])
	X1 = np.hstack([X1, X1_groups])

	X = np.concatenate([X0, X1], axis=0)
	Y = np.concatenate([Y0, Y1]).squeeze()

	

	for jj in range(n_repeats):
		
		if n_samples == None:
			curr_X = X
			curr_Y = Y
		else:
			## Subset data
			rand_idx = np.random.choice(np.arange(X.shape[0]), replace=False, size=n_samples)
			curr_X = X[rand_idx]
			curr_Y = Y[rand_idx]

		curr_X = np.hstack([curr_X[:, :n_genes], curr_X[:, -1:]])

		

		# rand_idx = np.random.choice(np.arange(X.shape[1]), replace=False, size=n_genes)
		# curr_X = curr_X[:, rand_idx]

		objective = lambda params: -log_marginal_likelihood(params, curr_X, curr_Y)

		num_params, predict, log_marginal_likelihood, unpack_kernel_params = \
					make_gp_funs(mg_rbf_covariance, num_cov_params=3) # params here are length scale, output scale, and a (group diff param)

		rs = npr.RandomState(0)
		init_params = 0.1 * rs.randn(num_params)

		def callback(params):
			pass

		res = minimize(value_and_grad(objective), init_params, jac=True,
							  method='CG', callback=callback)

		mean, cov_params, noise_scale = unpack_kernel_params(res.x)
		output_scale = cov_params[0]
		group_diff_param = np.exp(cov_params[1])
		lengthscales = cov_params[2:]
		curr_a = np.log(group_diff_param)
		results[jj, ii] = group_diff_param



		# #### Fit union GP
		# objective = lambda params: -log_marginal_likelihood(params, curr_X[:, :-1], curr_Y)

		# num_params, predict, log_marginal_likelihood, unpack_kernel_params = \
		# 			make_gp_funs(rbf_covariance, num_cov_params=2) # params here are length scale and output scale

		# rs = npr.RandomState(0)
		# init_params = 0.1 * rs.randn(num_params)

		# def callback(params):
		# 	pass

		# res = minimize(value_and_grad(objective), init_params, jac=True,
		# 					  method='CG', callback=callback)
		# import ipdb; ipdb.set_trace()

plt.figure(figsize=(7, 5))
results_df = pd.melt(pd.DataFrame(results, columns=tissue_labels))
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel("")
plt.ylabel(r"Estimated $a$")
plt.yscale("log")
plt.tight_layout()
plt.savefig("../plots/gtex_experiment_estimate_alpha.png")
plt.show()
import ipdb; ipdb.set_trace()

	# ## Fit union GP
	# gpr = GaussianProcessRegressor(kernel=RBF()) #, optimizer=None)
	# ll_union = gpr.fit(X[:, :-1], Y).log_marginal_likelihood()

	# ## Fit separate GPs
	# gpr = GaussianProcessRegressor(kernel=RBF()) #, optimizer=None)
	# ll0 = gpr.fit(X[:n0, :-1], Y[:n0]).log_marginal_likelihood()
	# ll1 = gpr.fit(X[n0:, :-1], Y[n0:]).log_marginal_likelihood()
	# ll_sep_gps = ll0 + ll1

	# ## Fit MGGP
	# alpha_list = [np.power(10, x * 1.0) for x in np.arange(-6, 6)]
	# ll_mggp_results = np.empty(len(alpha_list))
	# for jj, alpha in enumerate(alpha_list):

	# 	alpha = 1.0

	# 	## Fit MGGP
	# 	kernel = mgRBF(group_diff_param=alpha, length_scale=10.)
	# 	gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None)
		
	# 	# gpr = GaussianProcessRegressor(kernel=RBF()) #, optimizer=None)
	# 	# gpr = GaussianProcessRegressor(kernel=mgMatern(group_diff_param=alpha), optimizer=None)
	# 	ll_mggp = gpr.fit(X, Y).log_marginal_likelihood(eval_gradient=False)
	# 	# ll_mggp = gpr.fit(X, Y)
			

	# 	def obj_func(theta):
	# 		import ipdb; ipdb.set_trace()
	# 		return -gpr.log_marginal_likelihood(theta, clone_kernel=False)

		
	# 	opt_res = minimize(obj_func, kernel.theta, method="L-BFGS-B", jac=False, bounds=kernel.bounds)
	# 	ls_opt = np.exp(opt_res.x)
	# 	print(ls_opt)
	# 	import ipdb; ipdb.set_trace()
		
		
		
	# 	ll_mggp_results[jj] = ll_mggp
	# 	print(gpr.kernel_.length_scale)
	# 	print(ll_mggp)

		
# 	plt.subplot(1, len(data2_files), ii + 1)
# 	plt.plot(alpha_list, ll_mggp_results, label="MGGP")
# 	plt.axhline(ll_union, label="Union GP", linestyle="--", alpha=0.5, color="green")
# 	plt.axhline(ll_sep_gps, label="Separate GPs", linestyle="--", alpha=0.5, color="red")
# 	plt.xscale("log")
# 	plt.xlabel(r"$\alpha^2$")
# 	plt.ylabel(r"$\log p(Y)$")
# 	plt.title("Group 1: {}\nGroup 2: {}".format(group1_tissue, tissue_labels[ii]))
# 	plt.legend()
# 	plt.tight_layout()
# plt.savefig("../plots/gtex_experiment_rbf.png")
# plt.show()
# import ipdb; ipdb.set_trace()





