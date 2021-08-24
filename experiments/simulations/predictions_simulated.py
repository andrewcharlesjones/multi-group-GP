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
# from gaussian_process import make_gp_funs, 
from gaussian_process import GP, rbf_covariance, mg_rbf_covariance

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


n_repeats = 10
p = 1
noise_scale_true = 0.1
n0 = 50
n1 = 50
n = n0 + n1
frac_train = 0.5
# alpha_list = alpha_list = [np.power(10, x * 1.0) for x in np.arange(-5, 5)]
FIX_TRUE_PARAMS = False


def separated_gp():
	
	errors_mggp = np.empty(n_repeats)
	errors_separated_gp = np.empty(n_repeats)
	errors_union_gp = np.empty(n_repeats)

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

		all_idx = np.arange(n)
		train_idx = np.random.choice(all_idx, replace=False, size=int(frac_train * n))
		test_idx = np.setdiff1d(all_idx, train_idx)

		X_train = X[train_idx]
		X_test = X[test_idx]
		Y_train = Y[train_idx]
		Y_test = Y[test_idx]



		############################
		######### Fit MGGP #########
		############################

		mggp = GP(kernel=mg_rbf_covariance)
		mggp.fit(X_train, Y_train)
		preds_mean, _ = mggp.predict(X_test)
		curr_error = np.mean((Y_test - preds_mean)**2)
		errors_mggp[ii] = curr_error

		############################
		##### Fit separated GP #####
		############################

		X0_train = X_train[X_train[:, -1] == 0][:, :-1]
		Y0_train = Y_train[X_train[:, -1] == 0]
		X0_test = X_test[X_test[:, -1] == 0][:, :-1]
		Y0_test = Y_test[X_test[:, -1] == 0]

		sep_gp = GP(kernel=rbf_covariance)
		sep_gp.fit(X0_train, Y0_train)
		preds_mean, _ = sep_gp.predict(X0_test)
		curr_error0 = np.sum((Y0_test - preds_mean)**2)

		X1_train = X_train[X_train[:, -1] == 1][:, :-1]
		Y1_train = Y_train[X_train[:, -1] == 1]
		X1_test = X_test[X_test[:, -1] == 1][:, :-1]
		Y1_test = Y_test[X_test[:, -1] == 1]

		sep_gp = GP(kernel=rbf_covariance)
		sep_gp.fit(X1_train, Y1_train)
		preds_mean, _ = sep_gp.predict(X1_test)


		curr_error1 = np.sum((Y1_test - preds_mean)**2)


		errors_separated_gp[ii] = (curr_error0 + curr_error1) / (Y0_test.shape[0] + Y1_test.shape[0])

		############################
		####### Fit union GP #######
		############################

		union_gp = GP(kernel=rbf_covariance)
		union_gp.fit(X_train[:, :-1], Y_train)
		preds_mean, _ = union_gp.predict(X_test[:, :-1])
		curr_error = np.mean((Y_test - preds_mean)**2)
		errors_union_gp[ii] = curr_error


	results_df = pd.melt(pd.DataFrame({"MGGP": errors_mggp, "Separated GP": errors_separated_gp, "Union GP": errors_union_gp}))


	return results_df

def union_gp():

	errors_mggp = np.empty(n_repeats)
	errors_separated_gp = np.empty(n_repeats)
	errors_union_gp = np.empty(n_repeats)

	for ii in range(n_repeats):
	
		## Generate data
		X = np.random.uniform(low=-10, high=10, size=(n0 + n1, p))
		kernel = RBF()
		K_XX = kernel(X, X)
		Y = mvnpy.rvs(np.zeros(n0 + n1), K_XX) + np.random.normal(scale=noise_scale_true, size=n0 + n1)

		groups = np.concatenate([np.zeros((n0, 1)), np.ones((n1, 1))])
		X = np.hstack([X, groups])

		all_idx = np.arange(n)
		train_idx = np.random.choice(all_idx, replace=False, size=int(frac_train * n))
		test_idx = np.setdiff1d(all_idx, train_idx)

		X_train = X[train_idx]
		X_test = X[test_idx]
		Y_train = Y[train_idx]
		Y_test = Y[test_idx]



		############################
		######### Fit MGGP #########
		############################

		mggp = GP(kernel=mg_rbf_covariance)
		mggp.fit(X_train, Y_train)
		preds_mean, _ = mggp.predict(X_test)
		curr_error = np.mean((Y_test - preds_mean)**2)
		errors_mggp[ii] = curr_error

		############################
		##### Fit separated GP #####
		############################

		X0_train = X_train[X_train[:, -1] == 0][:, :-1]
		Y0_train = Y_train[X_train[:, -1] == 0]
		X0_test = X_test[X_test[:, -1] == 0][:, :-1]
		Y0_test = Y_test[X_test[:, -1] == 0]

		sep_gp = GP(kernel=rbf_covariance)
		sep_gp.fit(X0_train, Y0_train)
		preds_mean, _ = sep_gp.predict(X0_test)
		curr_error0 = np.sum((Y0_test - preds_mean)**2)

		X1_train = X_train[X_train[:, -1] == 1][:, :-1]
		Y1_train = Y_train[X_train[:, -1] == 1]
		X1_test = X_test[X_test[:, -1] == 1][:, :-1]
		Y1_test = Y_test[X_test[:, -1] == 1]

		sep_gp = GP(kernel=rbf_covariance)
		sep_gp.fit(X1_train, Y1_train)
		preds_mean, _ = sep_gp.predict(X1_test)


		curr_error1 = np.sum((Y1_test - preds_mean)**2)


		errors_separated_gp[ii] = (curr_error0 + curr_error1) / (Y0_test.shape[0] + Y1_test.shape[0])

		############################
		####### Fit union GP #######
		############################

		union_gp = GP(kernel=rbf_covariance)
		union_gp.fit(X_train[:, :-1], Y_train)
		preds_mean, _ = union_gp.predict(X_test[:, :-1])
		curr_error = np.mean((Y_test - preds_mean)**2)
		errors_union_gp[ii] = curr_error


	results_df = pd.melt(pd.DataFrame({"MGGP": errors_mggp, "Separated GP": errors_separated_gp, "Union GP": errors_union_gp}))
	# sns.boxplot(data=results_df, x="variable", y="value")
	# plt.show()

	# import ipdb; ipdb.set_trace()

	return results_df


def mggp():

	errors_mggp = np.empty(n_repeats)
	errors_separated_gp = np.empty(n_repeats)
	errors_union_gp = np.empty(n_repeats)

	

	for ii in range(n_repeats):
	
		## Generate data
		X = np.random.uniform(low=-10, high=10, size=(n0 + n1, p))
		

		groups = np.concatenate([np.zeros((n0, 1)), np.ones((n1, 1))])
		X = np.hstack([X, groups])
		K_XX = mg_rbf_covariance([1., a_true, 1.], X, X)
		Y = mvnpy.rvs(np.zeros(n0 + n1), K_XX) + np.random.normal(scale=noise_scale_true, size=n0 + n1)

		all_idx = np.arange(n)
		train_idx = np.random.choice(all_idx, replace=False, size=int(frac_train * n))
		test_idx = np.setdiff1d(all_idx, train_idx)

		X_train = X[train_idx]
		X_test = X[test_idx]
		Y_train = Y[train_idx]
		Y_test = Y[test_idx]


		############################
		######### Fit MGGP #########
		############################

		mggp = GP(kernel=mg_rbf_covariance)
		mggp.fit(X_train, Y_train)
		preds_mean, _ = mggp.predict(X_test)
		# import ipdb; ipdb.set_trace()
		curr_error = np.mean((Y_test - preds_mean)**2)
		errors_mggp[ii] = curr_error

		############################
		##### Fit separated GP #####
		############################

		X0_train = X_train[X_train[:, -1] == 0][:, :-1]
		Y0_train = Y_train[X_train[:, -1] == 0]
		X0_test = X_test[X_test[:, -1] == 0][:, :-1]
		Y0_test = Y_test[X_test[:, -1] == 0]

		sep_gp = GP(kernel=rbf_covariance)
		sep_gp.fit(X0_train, Y0_train)
		preds_mean, _ = sep_gp.predict(X0_test)
		curr_error0 = np.sum((Y0_test - preds_mean)**2)

		X1_train = X_train[X_train[:, -1] == 1][:, :-1]
		Y1_train = Y_train[X_train[:, -1] == 1]
		X1_test = X_test[X_test[:, -1] == 1][:, :-1]
		Y1_test = Y_test[X_test[:, -1] == 1]

		sep_gp = GP(kernel=rbf_covariance)
		sep_gp.fit(X1_train, Y1_train)
		preds_mean, _ = sep_gp.predict(X1_test)


		curr_error1 = np.sum((Y1_test - preds_mean)**2)


		errors_separated_gp[ii] = (curr_error0 + curr_error1) / (Y0_test.shape[0] + Y1_test.shape[0])

		############################
		####### Fit union GP #######
		############################

		union_gp = GP(kernel=rbf_covariance)
		union_gp.fit(X_train[:, :-1], Y_train)
		preds_mean, _ = union_gp.predict(X_test[:, :-1])
		curr_error = np.mean((Y_test - preds_mean)**2)
		errors_union_gp[ii] = curr_error


	results_df = pd.melt(pd.DataFrame({"MGGP": errors_mggp, "Separated GP": errors_separated_gp, "Union GP": errors_union_gp}))

	return results_df

if __name__ == '__main__':

	sep_gp_results = separated_gp()
	union_gp_results = union_gp()
	a_true = np.log(5e-1)
	mggp_results = mggp()

	sep_gp_results["method"] = ["Separated GP"] * sep_gp_results.shape[0]
	union_gp_results["method"] = ["Union GP"] * union_gp_results.shape[0]
	mggp_results["method"] = [r"MGGP, $a=" + str(round(np.exp(a_true), 1)) + "$"] * mggp_results.shape[0]

	all_results = pd.concat([sep_gp_results, union_gp_results, mggp_results], axis=0)
	cols = all_results.columns.values
	cols[cols == "variable"] = "Fitted model"
	all_results.columns = cols

	plt.figure(figsize=(8, 5))
	sns.boxplot(data=all_results, x="method", y="value", hue="Fitted model")
	plt.xlabel("Model from which data is generated")
	plt.ylabel("Prediction MSE")
	plt.yscale("log")
	plt.legend(fontsize=15, loc="upper center")
	plt.tight_layout()
	plt.savefig("../plots/prediction_simulation_experiment.png")
	plt.show()

	import ipdb; ipdb.set_trace()
	

