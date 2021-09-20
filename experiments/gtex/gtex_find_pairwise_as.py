# import numpy as np
import autograd.numpy as np
import autograd.numpy.random as npr
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys

sys.path.append("../../models")
from gaussian_process import (
    MGGP,
)
from kernels import (
    hierarchical_multigroup_kernel,
    multigroup_rbf_covariance,
)

import matplotlib

font = {"size": 15}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

DATA_DIR = "../../data/gtex"

tissue_labels = [
    "Anterior\ncingulate\ncortex",
    # "Frontal\ncortex",
    # "Cortex",
    # "Uterus",
    # "Vagina",
    "Breast",
    # "Tibial\nartery",
    # "Coronary\nartery",
    
]
data_prefixes = [
    "anterior_cingulate_cortex",
    # "frontal_cortex",
    # "cortex",
    # "uterus",
    # "vagina",
    "breast_mammary",
    # "tibial_artery",
    # "coronary_artery",
    
]
data_fnames = [x + "_expression.csv" for x in data_prefixes]
output_fnames = [x + "_ischemic_time.csv" for x in data_prefixes]

output_col = "TRISCHD"


n_repeats = 5
n_genes = 5
n_samples = 100
# n_samples = None
# n_genes = 2
# n_samples = 30
frac_train = 0.5

n_groups = len(tissue_labels)

between_group_dist = 1e0
within_group_dist = 1e-2
group_relationships = np.zeros((2, 2))

group_dist_mat = (
    group_relationships * within_group_dist
    + np.logical_not(group_relationships).astype(int) * between_group_dist
)
np.fill_diagonal(group_dist_mat, 0)


errors_mggp = np.empty(n_repeats)
errors_separated_gp = np.empty(n_repeats)
errors_union_gp = np.empty(n_repeats)
errors_hgp = np.empty(n_repeats)

errors_groupwise_mggp = np.empty((n_repeats, n_groups))
errors_groupwise_separated_gp = np.empty((n_repeats, n_groups))
errors_groupwise_union_gp = np.empty((n_repeats, n_groups))
errors_groupwise_hgp = np.empty((n_repeats, n_groups))


n_models = 2

for ii in range(n_repeats):

    X_list = []
    Y_list = []
    groups_list = []
    groups_ints = []
    for kk in range(n_groups):
        data = pd.read_csv(pjoin(DATA_DIR, data_fnames[kk]), index_col=0)
        output = pd.read_csv(pjoin(DATA_DIR, output_fnames[kk]), index_col=0)[
            output_col
        ]
        assert np.array_equal(data.index.values, output.index.values)

        curr_X, curr_Y = data.values, output.values

        if n_samples is None:
            rand_idx = np.arange(curr_X.shape[0])
        else:
            rand_idx = np.random.choice(
                np.arange(curr_X.shape[0]),
                replace=False,
                size=min(n_samples, curr_X.shape[0]),
            )

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

    ############################
    ######### Fit MGGP #########
    ############################

    mggp = MGGP(kernel=multigroup_rbf_covariance)


    mggp.fit(X, Y, X_groups, group_dist_mat)
    print("a: {}".format(np.exp(mggp.params[-2])))
    # import ipdb; ipdb.set_trace()
    
import ipdb

ipdb.set_trace()
