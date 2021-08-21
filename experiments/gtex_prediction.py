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
from gaussian_process import make_gp_funs, mg_rbf_covariance, rbf_covariance, mg_matern12_covariance, matern12_covariance
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
DATA_FILE1 = "anterior_cingulate_cortex_expression.csv"
# DATA_FILE2 = "breast_mammary_expression.csv"
# OUTPUT_FILE1 = "tibial_artery_ischemic_time.csv"
OUTPUT_FILE1 = "anterior_cingulate_cortex_ischemic_time.csv"
# OUTPUT_FILE2 = "breast_mammary_ischemic_time.csv"

# group1_tissue = "Tibial artery"
group1_tissue = "Anterior\ncingulate cortex"

data2_files = ["breast_mammary_expression.csv", "frontal_cortex_expression.csv"]
output2_files = ["breast_mammary_ischemic_time.csv", "frontal_cortex_ischemic_time.csv"]
# data2_files = ["tibial_artery_expression.csv", "frontal_cortex_expression.csv"]
# output2_files = ["tibial_artery_ischemic_time.csv", "frontal_cortex_ischemic_time.csv"]
# tissue_labels = ["Group 1: " + group1_tissue + "\nGroup 2: Breast", "Group 1: " + group1_tissue + "\nGroup 2: Coronary artery"]
tissue_labels = [r"$\textbf{Group 1}$: " + group1_tissue + "\n" + r"$\textbf{Group 2}$: Breast", r"$\textbf{Group 1}$: " + group1_tissue + "\n" + r"$\textbf{Group 2}$: Frontal cortex"]

# plt.figure(figsize=(7 * len(data2_files), 5))

n_repeats = 5
n_genes = 20
n_samples = 100
frac_train = 0.5

errors_mggp = np.empty((n_repeats, len(tissue_labels)))
errors_separated_gp = np.empty((n_repeats, len(tissue_labels)))
errors_union_gp = np.empty((n_repeats, len(tissue_labels)))

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

		## Subset data
		rand_idx = np.random.choice(np.arange(X.shape[0]), replace=False, size=n_samples)
		curr_X = X[rand_idx]
		curr_X = np.hstack([curr_X[:, :n_genes], curr_X[:, -1:]])
		curr_Y = Y[rand_idx]

		all_idx = np.arange(n_samples)
		train_idx = np.random.choice(all_idx, replace=False, size=int(frac_train * n_samples))
		test_idx = np.setdiff1d(all_idx, train_idx)

		X_train = curr_X[train_idx]
		X_test = curr_X[test_idx]
		Y_train = curr_Y[train_idx]
		Y_test = curr_Y[test_idx]
		
		

		############################
		######### Fit MGGP #########
		############################

		objective = lambda params: -log_marginal_likelihood(params, X_train, Y_train)

		num_params, predict, log_marginal_likelihood, unpack_kernel_params = \
					make_gp_funs(mg_matern12_covariance, num_cov_params=3) # params here are length scale, output scale, and a (group diff param)

		rs = npr.RandomState(0)
		init_params = 0.1 * rs.randn(num_params)

		def callback(params):
			pass

		res = minimize(value_and_grad(objective), init_params, jac=True,
							  method='CG', callback=callback)

		preds_mean, preds_cov = predict(res.x, X_train, Y_train, X_test)
		curr_error = np.mean((Y_test - preds_mean)**2)
		errors_mggp[jj, ii] = curr_error


		############################
		##### Fit separated GP #####
		############################

		# Build model and objective function.
		num_params, predict, log_marginal_likelihood, unpack_kernel_params = \
			make_gp_funs(matern12_covariance, num_cov_params=2) # params here are length scale and output scale

		X0_train = X_train[X_train[:, -1] == 0][:, :-1]
		Y0_train = Y_train[X_train[:, -1] == 0]
		X0_test = X_test[X_test[:, -1] == 0][:, :-1]
		Y0_test = Y_test[X_test[:, -1] == 0]

		objective = lambda params: -log_marginal_likelihood(params, X0_train, Y0_train)

		init_params = 0.1 * rs.randn(num_params)

		res = minimize(value_and_grad(objective), init_params, jac=True,
							  method='CG', callback=callback)
		
		preds_mean, preds_cov = predict(res.x, X0_train, Y0_train, X0_test)
		curr_error0 = np.sum((Y0_test - preds_mean)**2)

		X1_train = X_train[X_train[:, -1] == 1][:, :-1]
		Y1_train = Y_train[X_train[:, -1] == 1]
		X1_test = X_test[X_test[:, -1] == 1][:, :-1]
		Y1_test = Y_test[X_test[:, -1] == 1]
		objective = lambda params: -log_marginal_likelihood(params, X1_train, Y1_train)

		init_params = 0.1 * rs.randn(num_params)

		res = minimize(value_and_grad(objective), init_params, jac=True,
							  method='CG', callback=callback)
		
		preds_mean, preds_cov = predict(res.x, X1_train, Y1_train, X1_test)
		curr_error1 = np.sum((Y1_test - preds_mean)**2)


		errors_separated_gp[jj, ii] = (curr_error0 + curr_error1) / (Y0_test.shape[0] + Y1_test.shape[0])


		############################
		####### Fit union GP #######
		############################

		# Build model and objective function.
		num_params, predict, log_marginal_likelihood, unpack_kernel_params = \
			make_gp_funs(matern12_covariance, num_cov_params=2) # params here are length scale and output scale

		objective = lambda params: -log_marginal_likelihood(params, X_train[:, :-1], Y_train)

		init_params = 0.1 * rs.randn(num_params)

		res = minimize(value_and_grad(objective), init_params, jac=True,
							  method='CG', callback=callback)
		
		preds_mean, preds_cov = predict(res.x, X_train[:, :-1], Y_train, X_test[:, :-1])
		curr_error = np.mean((Y_test - preds_mean)**2)
		errors_union_gp[jj, ii] = curr_error



mggp_results_df = pd.melt(pd.DataFrame(errors_mggp, columns=tissue_labels))
separated_gp_results_df = pd.melt(pd.DataFrame(errors_separated_gp, columns=tissue_labels))
union_gp_results_df = pd.melt(pd.DataFrame(errors_union_gp, columns=tissue_labels))

separated_gp_results_df["model"] = ["Separated GP"] * separated_gp_results_df.shape[0]
union_gp_results_df["model"] = ["Union GP"] * union_gp_results_df.shape[0]
mggp_results_df["model"] = ["MGGP"] * mggp_results_df.shape[0]


results_df = pd.concat([mggp_results_df, separated_gp_results_df, union_gp_results_df])


plt.figure(figsize=(7, 5))
sns.boxplot(data=results_df, x="variable", y="value", hue="model")
plt.xlabel("")
plt.ylabel("Test MSE")
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("../plots/prediction_gtex_experiment.png")
plt.show()
import ipdb; ipdb.set_trace()







