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
from tqdm import tqdm
from matplotlib.pyplot import cm
from matplotlib.cm import get_cmap
from scipy.stats import pearsonr

import sys

sys.path.append("../../models")
from gaussian_process import (
    MGGP,
    GP,
    HGP,
)
from kernels import (
    hierarchical_multigroup_kernel,
    rbf_covariance,
    multigroup_rbf_covariance,
)

import matplotlib

font = {"size": 15}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

DATA_DIR = "../../data/gtex"

tissue_labels = [
    "Anterior\ncingulate\ncortex",
    "Frontal\ncortex",
    # "Cortex",
    "Breast",
    # "Tibial\nartery",
    # "Coronary\nartery",
    # "Uterus",
    # "Vagina",
]
data_prefixes = [
    "anterior_cingulate_cortex",
    "frontal_cortex",
    # "cortex",
    "breast_mammary",
    # "tibial_artery",
    # "coronary_artery",
    # "uterus",
    # "vagina",
]
data_fnames = [x + "_expression.csv" for x in data_prefixes]
output_fnames = [x + "_ischemic_time.csv" for x in data_prefixes]

output_col = "TRISCHD"


n_repeats = 2
# gene_idx = 3
n_samples = None
n_test = 20

n_groups = len(tissue_labels)

between_group_dist = 1e0
within_group_dist = 1e-1
group_relationships = np.array(
    [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
)
# group_relationships = np.array(
#     [
#         [1, 1, 1, 0, 0, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 1, 0, 0],
#         [0, 0, 0, 0, 1, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 1],
#         [0, 0, 0, 0, 0, 0, 1, 1],
#     ]
# )

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


X_list = []
Y_list = []
groups_list = []
groups_ints = []
for kk in range(n_groups):
    data = pd.read_csv(pjoin(DATA_DIR, data_fnames[kk]), index_col=0)
    output = pd.read_csv(pjoin(DATA_DIR, output_fnames[kk]), index_col=0)[output_col]
    # positive_it_idx = np.where(output.values > 0)[0]
    # data = data.iloc[positive_it_idx, :]
    # output = output.iloc[positive_it_idx]
    assert np.array_equal(data.index.values, output.index.values)

    if kk == 0:
        sorted_idx = np.argsort(data.corrwith(output).values)
        gene_names = data.columns.values
        gene_idx = sorted_idx[0]
        gene_name = gene_names[gene_idx]
        print(gene_name)

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
Y_mean = Y.mean(0)
Y_std = Y.std(0)
Y = (Y - Y_mean) / Y_std

X = X[:, np.array(gene_idx)]
X = np.expand_dims(X, axis=1)

############################
######### Fit MGGP #########
############################

mggp = MGGP(kernel=multigroup_rbf_covariance)

mggp.fit(X, Y, X_groups, group_dist_mat)


X_test_one_group = np.linspace(np.min(X), np.max(X), n_test)
X_test_one_group = np.expand_dims(X_test_one_group, 1)
X_test = []
X_groups_test = []
groups_ints_test = []
for kk in range(n_groups):
    curr_group_one_hot = np.zeros(n_groups)
    curr_group_one_hot[kk] = 1
    curr_groups = np.repeat([curr_group_one_hot], n_test, axis=0)
    X_test.append(X_test_one_group)
    X_groups_test.append(curr_groups)

    groups_ints_test.append(np.repeat(kk, n_test))

X_test = np.concatenate(X_test, axis=0)
X_groups_test = np.concatenate(X_groups_test, axis=0)
groups_ints_test = np.concatenate(groups_ints_test)

preds_mean, _ = mggp.predict(X_test, X_groups_test)

plt.figure(figsize=(9, 6))
cmap = get_cmap("tab10")

for kk in range(n_groups):
    plt.scatter(
        Y[groups_ints == kk] * Y_std + Y_mean,
        X[groups_ints == kk],
        color=cmap.colors[kk],
        alpha=0.6,
    )
    plt.plot(
        preds_mean[groups_ints_test == kk] * Y_std + Y_mean,
        X_test[groups_ints_test == kk],
        color=cmap.colors[kk],
        label=tissue_labels[kk],
        linewidth=5,
    )

plt.title(gene_name)


pearsonr(X[groups_ints == 0].squeeze(), Y[groups_ints == 0])[0]
pearsonr(X[groups_ints == 1].squeeze(), Y[groups_ints == 1])[0]
pearsonr(X[groups_ints == 2].squeeze(), Y[groups_ints == 2])[0]

plt.legend()
plt.ylabel("Gene expression")
plt.xlabel("Ischemic time")
plt.tight_layout()
plt.savefig("../../plots/gtex_one_gene_relationship_{}.png".format(gene_name))
plt.show()

import ipdb

ipdb.set_trace()
