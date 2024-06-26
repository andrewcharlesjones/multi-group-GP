import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Sum
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
from scipy.optimize import minimize
from sklearn.decomposition import PCA

import sys

# from multigroupGP import GP, multigroup_rbf_kernel, rbf_kernel
from multigroupGP import GP, MultiGroupRBF, RBF

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

DATA_DIR = "../../data/gtex"
DATA_FILE1 = "Brain_Frontal_Cortex_(BA9)_expression.csv"
OUTPUT_FILE1 = "Brain_Frontal_Cortex_(BA9)_ischemic_time.csv"

group1_tissue = "Anterior\ncingulate cortex"

data_prefixes = [
    "Brain_Anterior_cingulate_cortex_(BA24)",
    # "Brain_Frontal_Cortex_(BA9)",
    # "cortex",
    # "Breast_Mammary_Tissue",
    # "tibial_artery",
    "Artery_Coronary",
    # "uterus",
    # "vagina",
]
data2_files = [x + "_expression.csv" for x in data_prefixes]
output2_files = [x + "_ischemic_time.csv" for x in data_prefixes]

tissue_labels = [
    r"$\textbf{Group 1}$: "
    + group1_tissue
    + "\n"
    + r"$\textbf{Group 2}$: Frontal cortex",
    r"$\textbf{Group 1}$: "
    + group1_tissue
    + "\n"
    + r"$\textbf{Group 2}$: Coronary artery",
]


output_col = "TRISCHD"


n_genes = 20
a_list = [np.power(10, x * 1.0) for x in np.arange(-6, 6)]
group_dist_mat = np.array([[0, 1], [1, 0]])
plt.figure(figsize=(14, 7))

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

    X = X[:, :n_genes]

    X = PCA(n_components=n_genes).fit_transform(X)

    X = (X - X.mean(0)) / X.std(0)
    Y = (Y - Y.mean(0)) / Y.std(0)

    X0 = X[np.where(X_groups == 0)[1].astype(bool)]
    X1 = X[np.where(X_groups == 1)[1].astype(bool)]
    Y0 = Y[np.where(X_groups == 0)[1].astype(bool)]
    Y1 = Y[np.where(X_groups == 1)[1].astype(bool)]

    ## MGGP
    LL_list = []
    for curr_a in a_list:

        kernel_params = [np.log(1.0), np.log(curr_a), np.log(1.0)]
        kernel = MultiGroupRBF(
            amplitude=1.0,
            group_diff_param=curr_a,
            lengthscale=1.0,
            # kernel_params=kernel_params,
        )
        curr_K_XX = kernel(
            x1=X,
            x2=X,
            groups1=X_groups,
            groups2=X_groups,
            group_distances=group_dist_mat,
            log_params=False,
        )
        curr_LL = mvn.logpdf(Y, np.zeros(len(X)), curr_K_XX + np.eye(len(X)))
        LL_list.append(curr_LL)

    plt.subplot(1, 2, ii + 1)
    plt.plot(a_list, LL_list, color="black", label="Multi-Group")

    ## Union GP
    # kernel_params = [np.log(1.0), np.log(1.0)]
    kernel = RBF(amplitude=1.0, lengthscale=1.0)
    curr_K_XX = kernel(X, log_params=False)
    # import ipdb; ipdb.set_trace()
    ll_union = mvn.logpdf(Y, np.zeros(len(X)), curr_K_XX + np.eye(len(X)))
    plt.axhline(ll_union, label="Union", linestyle="--", color="green")

    ## Separate GPs
    ll_sep = 0
    # curr_K_X0X0 = RBF(kernel_params, X0, X0)
    curr_K_X0X0 = kernel(X0, log_params=False)
    ll_sep += mvn.logpdf(Y0, np.zeros(len(X0)), curr_K_X0X0 + np.eye(len(X0)))
    # curr_K_X1X1 = RBF(kernel_params, X1, X1)
    curr_K_X1X1 = kernel(X1, log_params=False)
    ll_sep += mvn.logpdf(Y1, np.zeros(len(X1)), curr_K_X1X1 + np.eye(len(X1)))
    plt.axhline(ll_sep, label="Separate", linestyle="--", color="red")

    plt.xlabel(r"$a$")
    plt.ylabel(r"$\log p(Y)$")
    plt.xscale("log")
    plt.title(tissue_labels[ii])
    plt.legend()
plt.tight_layout()
plt.savefig("../../plots/gtex_two_group_a_range.png", dpi=300)
plt.show()
import ipdb

ipdb.set_trace()
