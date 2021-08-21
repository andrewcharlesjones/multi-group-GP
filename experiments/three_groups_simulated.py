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
from gaussian_process import (
	make_gp_funs,
	mg_rbf_covariance,
	rbf_covariance,
	threegroup_rbf_covariance,
	MGGP,
	GP,
)

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


n_repeats = 2
p = 1
noise_scale_true = 0.1
# n0 = 30
n1 = 30
n2 = 30
n_groups = 3
# n = n0 + n1 + n2
frac_train = 0.5
FIX_TRUE_PARAMS = False
n0_list = [5, 10, 30, 50]


def experiment():

	errors_mggp = np.empty((n_repeats, len(n0_list)))
	errors_separated_gp = np.empty((n_repeats, len(n0_list)))
	errors_union_gp = np.empty((n_repeats, len(n0_list)))

	for ii in range(n_repeats):

		for jj, n0 in enumerate(n0_list):
			n = n0 + n1 + n2
			## Generate data from MGGP

			X = np.random.uniform(low=-10, high=10, size=(n, p))

			X0_groups = np.repeat([[1, 0, 0]], n0, axis=0)
			X1_groups = np.repeat([[0, 1, 0]], n1, axis=0)
			X2_groups = np.repeat([[0, 0, 1]], n2, axis=0)
			X_groups = np.concatenate([X0_groups, X1_groups, X2_groups], axis=0)

			g12_dist = 1e-1
			g13_dist = 5.0
			g23_dist = 5.0
			a = 1e0
			group_dist_mat = np.array(
				[
					[0.0, g12_dist, g13_dist],
					[g12_dist, 0.0, g23_dist],
					[g13_dist, g23_dist, 0.0],
				]
			)
			K_XX = threegroup_rbf_covariance(
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

			all_idx = np.arange(n)
			train_idx = np.random.choice(all_idx, replace=False, size=int(frac_train * n))
			test_idx = np.setdiff1d(all_idx, train_idx)

			X_train = X[train_idx]
			X_test = X[test_idx]
			Y_train = Y[train_idx]
			Y_test = Y[test_idx]
			X_groups_train = X_groups[train_idx]
			X_groups_test = X_groups[test_idx]

			############################
			######### Fit MGGP #########
			############################

			mggp = MGGP(kernel=threegroup_rbf_covariance)
			mggp.fit(X_train, Y_train, X_groups_train, group_dist_mat)
			preds_mean, _ = mggp.predict(X_test, X_groups_test)

			# curr_error = np.mean((Y_test - preds_mean) ** 2)
			group0_idx = X_groups_test[:, 0] == 1
			curr_error = np.mean((Y_test[group0_idx] - preds_mean[group0_idx]) ** 2)
			errors_mggp[ii, jj] = curr_error

			############################
			##### Fit separated GP #####
			############################

			sum_error_sep_gp = 0
			for groupnum in range(n_groups):
				curr_X_train = X_train[X_groups_train[:, groupnum] == 1]
				curr_Y_train = Y_train[X_groups_train[:, groupnum] == 1]
				curr_X_test = X_test[X_groups_test[:, groupnum] == 1]
				curr_Y_test = Y_test[X_groups_test[:, groupnum] == 1]

				sep_gp = GP(kernel=rbf_covariance)
				sep_gp.fit(curr_X_train, curr_Y_train)
				preds_mean, _ = sep_gp.predict(curr_X_test)
				curr_error_sep_gp = np.sum((curr_Y_test - preds_mean) ** 2)

				if groupnum == 0:
					sum_error_sep_gp += curr_error_sep_gp
					# import ipdb; ipdb.set_trace()

			# errors_separated_gp[ii, jj] = sum_error_sep_gp / Y_test.shape[0]
			errors_separated_gp[ii, jj] = sum_error_sep_gp / sum(group0_idx.astype(int))

			############################
			####### Fit union GP #######
			############################

			union_gp = GP(kernel=rbf_covariance)
			union_gp.fit(X_train, Y_train)
			preds_mean, _ = union_gp.predict(X_test)
			# curr_error = np.mean((Y_test - preds_mean) ** 2)
			curr_error = np.mean((Y_test[group0_idx] - preds_mean[group0_idx]) ** 2)
			errors_union_gp[ii, jj] = curr_error

	

	errors_mggp_df = pd.melt(pd.DataFrame(errors_mggp, columns=n0_list))
	errors_mggp_df['model'] = ["MGGP"] * errors_mggp_df.shape[0]
	errors_separated_gp_df = pd.melt(pd.DataFrame(errors_separated_gp, columns=n0_list))
	errors_separated_gp_df['model'] = ["Separated GP"] * errors_separated_gp_df.shape[0]
	errors_union_gp_df = pd.melt(pd.DataFrame(errors_union_gp, columns=n0_list))
	errors_union_gp_df['model'] = ["Union GP"] * errors_union_gp_df.shape[0]

	results_df = pd.concat([errors_mggp_df, errors_separated_gp_df, errors_union_gp_df], axis=0)



	# import ipdb; ipdb.set_trace()

	# results_df = pd.melt(
	# 	pd.DataFrame(
	# 		{
	# 			"MGGP": errors_mggp,
	# 			"Separated GP": errors_separated_gp,
	# 			"Union GP": errors_union_gp,
	# 		}
	# 	)
	# )

	return results_df, X, Y, X_groups


if __name__ == "__main__":

	results_df, X, Y, X_groups = experiment()

	plt.figure(figsize=(14, 5))

	plt.subplot(121)
	for ii in range(n_groups):
		curr_idx = X_groups[:, ii] == 1
		plt.scatter(X[curr_idx], Y[curr_idx], label="Group {}".format(ii + 1))
	plt.xlabel(r"$X$")
	plt.ylabel(r"$Y$")
	plt.title("Data")
	plt.legend()

	plt.subplot(122)
	# sns.boxplot(data=results_df, x="variable", y="value")
	sns.lineplot(data=results_df, x="variable", y="value", hue="model")
	plt.ylabel("Prediction MSE")
	plt.xlabel(r"$n$, group 1")
	plt.tight_layout()
	plt.savefig("../plots/three_group_simulated_prediction.png")
	plt.show()

	import ipdb

	ipdb.set_trace()
