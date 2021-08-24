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
from sklearn.decomposition import PCA

import sys
sys.path.append("../../models")
from gaussian_process import GP, HGP, MGGP, multigroup_rbf_covariance, rbf_covariance

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

DATA_DIR = "../../data/gtex"
# DATA_FILE1 = "tibial_artery_expression.csv"
DATA_FILE1 = "frontal_cortex_expression.csv"
# DATA_FILE2 = "breast_mammary_expression.csv"
# OUTPUT_FILE1 = "tibial_artery_ischemic_time.csv"
OUTPUT_FILE1 = "frontal_cortex_ischemic_time.csv"
# OUTPUT_FILE2 = "breast_mammary_ischemic_time.csv"

# group1_tissue = "Tibial artery"
group1_tissue = "Anterior\ncingulate cortex"

data2_files = ["breast_mammary_expression.csv", "anterior_cingulate_cortex_expression.csv"]
output2_files = ["breast_mammary_ischemic_time.csv", "anterior_cingulate_cortex_ischemic_time.csv"]
# tissue_labels = ["Group 1: " + group1_tissue + "\nGroup 2: Breast", "Group 1: " + group1_tissue + "\nGroup 2: Coronary artery"]
tissue_labels = [r"$\textbf{Group 1}$: " + group1_tissue + "\n" + r"$\textbf{Group 2}$: Breast", r"$\textbf{Group 1}$: " + group1_tissue + "\n" + r"$\textbf{Group 2}$: Frontal cortex"]


output_col = "TRISCHD"


n_repeats = 5
n_genes = 5
n_samples = 100
results = np.empty((n_repeats, len(tissue_labels)))

for ii in range(len(data2_files)):

	DATA_FILE2 = data2_files[ii]
	OUTPUT_FILE2 = output2_files[ii]

	## Load data
	data1 = pd.read_csv(pjoin(DATA_DIR, DATA_FILE1), index_col=0)
	data2 = pd.read_csv(pjoin(DATA_DIR, DATA_FILE2), index_col=0)

	output1 = pd.read_csv(pjoin(DATA_DIR, OUTPUT_FILE1), index_col=0)[output_col]
	output2 = pd.read_csv(pjoin(DATA_DIR, OUTPUT_FILE2), index_col=0)[output_col]

	assert np.array_equal(data1.index.values, output1.index.values)
	assert np.array_equal(data2.index.values, output2.index.values)

	X0, Y0 = data1.values, output1.values
	X1, Y1 = data2.values, output2.values

	n0, n1 = X0.shape[0], X1.shape[0]

	X0_groups = np.repeat([[1, 0]], n0, axis=0)
	X1_groups = np.repeat([[0, 1]], n1, axis=0)
	X_groups = np.concatenate([X0_groups, X1_groups], axis=0)

	X = np.concatenate([X0, X1], axis=0)
	Y = np.concatenate([Y0, Y1]).squeeze()

	X = np.log(X + 1)
	X = (X - X.mean(0)) / X.std(0)
	Y = (Y - Y.mean(0)) / Y.std(0)

	X = PCA(n_components=n_genes).fit_transform(X)
	

	for jj in range(n_repeats):
		
		if n_samples == None:
			rand_idx = np.arange(X.shape[0])
		else:
			## Subset data
			rand_idx = np.random.choice(np.arange(X.shape[0]), replace=False, size=n_samples)

		curr_X = X[rand_idx]
		curr_Y = Y[rand_idx]
		curr_X_groups = X_groups[rand_idx]

		# curr_X = curr_X[:, :n_genes]

		############################
		######### Fit MGGP #########
		############################

		mggp = MGGP(kernel=multigroup_rbf_covariance)
		mggp.fit(curr_X, curr_Y, curr_X_groups)
		curr_a = np.exp(mggp.params[3])
		results[jj, ii] = curr_a

plt.figure(figsize=(7, 5))
results_df = pd.melt(pd.DataFrame(results, columns=tissue_labels))
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel("")
plt.ylabel(r"Estimated $a$")
plt.yscale("log")
plt.tight_layout()
plt.savefig("../../plots/gtex_experiment_estimate_alpha.png")
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





