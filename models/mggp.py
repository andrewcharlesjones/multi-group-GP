import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
from scipy.stats import multivariate_normal as mvn
import sys
sys.path.append("../kernels")
from mgRBF import mgRBF
from RBF import RBF

if __name__ == "__main__":

	## Generate data from independent GP for each group

	n0 = 100
	n1 = 90
	p = 1
	X0 = np.random.normal(size=(n0, p))
	X1 = np.random.normal(size=(n1, p))
	kernel = RBF()
	K_X0X0 = kernel(X0, X0)
	K_X1X1 = kernel(X1, X1)
	Y0 = mvn.rvs(np.zeros(n0), K_X0X0)
	Y1 = mvn.rvs(np.zeros(n1), K_X1X1)
	
	## Fit independent GPs
	gpr = GaussianProcessRegressor(kernel=RBF(), optimizer=None)
	ll0 = gpr.fit(X0, Y0).log_marginal_likelihood()
	ll1 = gpr.fit(X1, Y1).log_marginal_likelihood()
	ll_independent = ll0 + ll1
	
	## Fit MGGP
	X0_groups = np.zeros((n0, 1))
	X1_groups = np.ones((n1, 1))
	X0 = np.hstack([X0, X0_groups])
	X1 = np.hstack([X1, X1_groups])
	X = np.concatenate([X0, X1], axis=0)
	Y = np.concatenate([Y0, Y1])
	gpr = GaussianProcessRegressor(kernel=mgRBF(group_diff_param=1e3), optimizer=None)
	ll_mggp = gpr.fit(X, Y).log_marginal_likelihood()

	print("LL GP: {0:10}".format(round(ll_independent, 2)))
	print("LL MGGP: {0:10}".format(round(ll_mggp, 2)))

	import ipdb; ipdb.set_trace()
	# gpr.

