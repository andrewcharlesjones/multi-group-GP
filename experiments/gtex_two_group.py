import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin

import sys
sys.path.append("../kernels")
from mgRBF import mgRBF
from RBF import RBF
from RBF_groupwise import RBF_groupwise

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

DATA_DIR = "../data/gtex"
DATA_FILE1 = "tibial_artery_expression.csv"
# DATA_FILE2 = "breast_mammary_expression.csv"
OUTPUT_FILE1 = "tibial_artery_ischemic_time.csv"
# OUTPUT_FILE2 = "breast_mammary_ischemic_time.csv"

group1_tissue = "Tibial artery"

data2_files = ["breast_mammary_expression.csv", "coronary_artery_expression.csv"]
output2_files = ["breast_mammary_ischemic_time.csv", "coronary_artery_ischemic_time.csv"]
tissue_labels = ["Breast", "Coronary artery"]

plt.figure(figsize=(7 * len(data2_files), 5))
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
	# import ipdb; ipdb.set_trace()
	X = np.concatenate([X0, X1], axis=0)
	Y = np.concatenate([Y0, Y1])

	## Fit union GP
	gpr = GaussianProcessRegressor(kernel=RBF(), optimizer=None)
	ll_union = gpr.fit(X[:, :-1], Y).log_marginal_likelihood()

	## Fit separate GPs
	gpr = GaussianProcessRegressor(kernel=RBF(), optimizer=None)
	ll0 = gpr.fit(X[:n0, :-1], Y[:n0]).log_marginal_likelihood()
	ll1 = gpr.fit(X[n0:, :-1], Y[n0:]).log_marginal_likelihood()
	ll_sep_gps = ll0 + ll1

	## Fit MGGP
	alpha_list = [np.power(10, x * 1.0) for x in np.arange(-6, 6)]
	ll_mggp_results = np.empty(len(alpha_list))
	for jj, alpha in enumerate(alpha_list):

		## Fit MGGP
		gpr = GaussianProcessRegressor(kernel=mgRBF(group_diff_param=alpha), optimizer=None)
		ll_mggp = gpr.fit(X, Y).log_marginal_likelihood()
		ll_mggp_results[jj] = ll_mggp

		
	plt.subplot(1, len(data2_files), ii + 1)
	plt.plot(alpha_list, ll_mggp_results, label="MGGP")
	plt.axhline(ll_union, label="Union GP", linestyle="--", alpha=0.5, color="green")
	plt.axhline(ll_sep_gps, label="Separate GPs", linestyle="--", alpha=0.5, color="red")
	plt.xscale("log")
	plt.xlabel(r"$\alpha^2$")
	plt.ylabel(r"$\log p(Y)$")
	plt.title("Group 1: {}\nGroup 2: {}".format(group1_tissue, tissue_labels[ii]))
	plt.legend()
	plt.tight_layout()
plt.savefig("../plots/gtex_experiment.png")
plt.show()
import ipdb; ipdb.set_trace()





