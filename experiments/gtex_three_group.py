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
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../models")
from gaussian_process import make_gp_funs, mg_rbf_covariance, rbf_covariance, mg_matern12_covariance, matern12_covariance, multigroup_rbf_covariance, MGGP, GP
sys.path.append("../kernels")
from mgRBF import mgRBF
from mgMatern import mgMatern
from RBF import RBF
from RBF_groupwise import RBF_groupwise

import matplotlib
font = {"size": 15}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

DATA_DIR = "../data/gtex"

tissue_labels = [
	"Anterior\ncingulate\ncortex",
	"Frontal\ncortex",
	"Cortex",
	"Breast",
	"Tibial\nartery",
	"Coronary\nartery",
	"Uterus",
	"Vagina"
]
data_prefixes = [
	"anterior_cingulate_cortex",
	"frontal_cortex",
	"cortex",
	"breast_mammary",
	"tibial_artery",
	"coronary_artery",
	"uterus",
	"vagina"
]
data_fnames = [
	x + "_expression.csv" for x in data_prefixes
]
output_fnames = [
	x + "_ischemic_time.csv" for x in data_prefixes
]

output_col = "TRISCHD"


n_repeats = 2
n_genes = 10
n_samples = 100
frac_train = 0.5

n_groups = len(tissue_labels)

between_group_dist = 1e0
within_group_dist = 1e-1
# group_relationships = np.array(
# 	[
# 		[1, 1, 0],
# 		[1, 1, 0],
# 		[0, 0, 0],
# 	]
# )
group_relationships = np.array(
	[
		[1, 1, 1, 0, 0, 0, 0, 0],
		[1, 1, 1, 0, 0, 0, 0, 0],
		[1, 1, 1, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 1, 1, 0, 0],
		[0, 0, 0, 0, 1, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 1, 1],
		[0, 0, 0, 0, 0, 0, 1, 1],
	]
)

group_dist_mat = group_relationships * within_group_dist + np.logical_not(group_relationships).astype(int) * between_group_dist
np.fill_diagonal(group_dist_mat, 0)

	
errors_mggp = np.empty(n_repeats)
errors_separated_gp = np.empty(n_repeats)
errors_union_gp = np.empty(n_repeats)

errors_groupwise_mggp = np.empty((n_repeats, n_groups))
errors_groupwise_separated_gp = np.empty((n_repeats, n_groups))
errors_groupwise_union_gp = np.empty((n_repeats, n_groups))

for ii in range(n_repeats):

	X_list = []
	Y_list = []
	groups_list = []
	groups_ints = []
	for kk in range(n_groups):
		data = pd.read_csv(pjoin(DATA_DIR, data_fnames[kk]), index_col=0)
		output = pd.read_csv(pjoin(DATA_DIR, output_fnames[kk]), index_col=0)[output_col]
		assert np.array_equal(data.index.values, output.index.values)

		curr_X, curr_Y = data.values, output.values

		if kk == 0:
			rand_idx = np.random.choice(np.arange(curr_X.shape[0]), replace=False, size=10)
		else:
			## Subset data
			rand_idx = np.random.choice(np.arange(curr_X.shape[0]), replace=False, size=n_samples)

		# rand_idx = np.random.choice(np.arange(curr_X.shape[0]), replace=False, size=min(n_samples, curr_X.shape[0]))

		curr_X = curr_X[rand_idx]
		curr_Y = curr_Y[rand_idx]

		curr_n = curr_X.shape[0]

		curr_group_one_hot = np.zeros(n_groups)
		curr_group_one_hot[kk] = 1
		curr_groups = np.repeat([curr_group_one_hot], curr_n, axis=0)

		X_list.append(curr_X)
		Y_list.append(curr_Y)
		groups_list.append(curr_groups)

		groups_ints.append(np.repeat(kk, curr_X.shape[0]))

	X_groups = np.concatenate(groups_list, axis=0)
	groups_ints = np.concatenate(groups_ints)


	X = np.concatenate(X_list, axis=0)
	Y = np.concatenate(Y_list).squeeze()
	ntotal = X.shape[0]

	X = np.log(X + 1)
	X = (X - X.mean(0)) / X.std(0)

	# Y = np.log(Y + 1)
	Y = (Y - Y.mean(0)) / Y.std(0)

	pca = PCA(n_components=n_genes)
	X = pca.fit_transform(X)

	all_idx = np.arange(ntotal)
	train_idx, test_idx, _, _ = train_test_split(all_idx, all_idx, stratify=groups_ints, test_size=frac_train, random_state=9)

	X_train = X[train_idx]
	X_test = X[test_idx]
	Y_train = Y[train_idx]
	Y_test = Y[test_idx]

	X_groups_train = X_groups[train_idx]
	X_groups_test = X_groups[test_idx]

	############################
	######### Fit MGGP #########
	############################

	mggp = MGGP(kernel=multigroup_rbf_covariance)
	mggp.fit(X_train, Y_train, X_groups_train, group_dist_mat)
	preds_mean, _ = mggp.predict(X_test, X_groups_test)
	curr_error = np.mean((Y_test - preds_mean) ** 2)
	errors_mggp[ii] = curr_error
	# print("MGGP: {}".format(curr_error))

	for groupnum in range(n_groups):
		curr_idx = X_groups_test[:, groupnum] == 1
		curr_error = np.mean((Y_test[curr_idx] - preds_mean[curr_idx]) ** 2)
		errors_groupwise_mggp[ii, groupnum] = curr_error

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
		sum_error_sep_gp += curr_error_sep_gp

		errors_groupwise_separated_gp[ii, groupnum] = np.mean((curr_Y_test - preds_mean) ** 2)

	errors_separated_gp[ii] = sum_error_sep_gp / Y_test.shape[0]


	############################
	####### Fit union GP #######
	############################

	union_gp = GP(kernel=rbf_covariance)
	union_gp.fit(X_train, Y_train)
	preds_mean, _ = union_gp.predict(X_test)
	curr_error = np.mean((Y_test - preds_mean) ** 2)
	errors_union_gp[ii] = curr_error
	print("Union: {}".format(curr_error))


	for groupnum in range(n_groups):
		curr_idx = X_groups_test[:, groupnum] == 1
		curr_error = np.mean((Y_test[curr_idx] - preds_mean[curr_idx]) ** 2)
		errors_groupwise_union_gp[ii, groupnum] = curr_error



results_df = pd.melt(pd.DataFrame({"MGGP": errors_mggp, "Union GP": errors_union_gp, "Separate GPs": errors_separated_gp}))


plt.figure(figsize=(14, 5))
plt.subplot(121)
sns.boxplot(data=results_df, x="variable", y="value")
plt.title("Total error")
plt.xlabel("")
plt.ylabel("Test MSE")
plt.tight_layout()

results_groupwise_mggp_df = pd.melt(pd.DataFrame(errors_groupwise_mggp, columns=tissue_labels))
results_groupwise_mggp_df['model'] = ["MGGP"] * results_groupwise_mggp_df.shape[0]
results_groupwise_union_df = pd.melt(pd.DataFrame(errors_groupwise_union_gp, columns=tissue_labels))
results_groupwise_union_df['model'] = ["Union GP"] * results_groupwise_union_df.shape[0]
results_groupwise_sep_df = pd.melt(pd.DataFrame(errors_groupwise_separated_gp, columns=tissue_labels))
results_groupwise_sep_df['model'] = ["Separated GPs"] * results_groupwise_sep_df.shape[0]

results_df_groupwise = pd.concat([results_groupwise_mggp_df, results_groupwise_union_df, results_groupwise_sep_df], axis=0)

plt.subplot(122)
g = sns.boxplot(data=results_df_groupwise, x="variable", y="value", hue="model")
plt.title("Group-wise error")
plt.xlabel("")
plt.ylabel("Test MSE")
g.legend_.set_title(None)
plt.tight_layout()
plt.savefig("../plots/prediction_gtex_experiment_multigroup.png")
plt.show()
import ipdb; ipdb.set_trace()







