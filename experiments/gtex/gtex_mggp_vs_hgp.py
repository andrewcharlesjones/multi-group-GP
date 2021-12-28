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

# sys.path.append("../../models")
# from gaussian_process import (
    
# )
from multigroupGP import (
    GP,
    HGPKernel,
    RBF,
    MultiGroupRBF,
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
    # "Uterus",
    # "Vagina",
    "Breast",
    # "Tibial\nartery",
    # "Coronary\nartery",
]
data_prefixes = [
    "Brain_Anterior_cingulate_cortex_(BA24)",
    "Brain_Frontal_Cortex_(BA9)",
    # "Brain_Cortex",
    "Breast_Mammary_Tissue",
    # "Artery_Tibial",
    # "Artery_Coronary",
    # "Uterus",
    # "Vagina",
]
data_fnames = [x + "_expression.csv" for x in data_prefixes]
output_fnames = [x + "_ischemic_time.csv" for x in data_prefixes]

output_col = "TRISCHD"


n_repeats = 5

PCA_REDUCE = False
n_components = 10
n_samples = 150
# n_samples = None
# n_genes = 2
# n_samples = 30
frac_train = 0.5

n_groups = len(tissue_labels)

between_group_dist = 1e0
within_group_dist = 1e-4
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


n_models = 2
pbar = tqdm(total=n_repeats * n_models)

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

        # if kk in [0, 1]:
        # 	rand_idx = np.random.choice(np.arange(curr_X.shape[0]), replace=False, size=10)
        # else:
        # 	## Subset data
        # 	rand_idx = np.random.choice(np.arange(curr_X.shape[0]), replace=False, size=min(n_samples, curr_X.shape[0]))

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

    if PCA_REDUCE:
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)

    all_idx = np.arange(ntotal)
    train_idx, test_idx, _, _ = train_test_split(
        all_idx, all_idx, stratify=groups_ints, test_size=frac_train, random_state=9
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    X_groups_train = X_groups[train_idx]
    X_groups_test = X_groups[test_idx]

    groups_ints_train = groups_ints[train_idx]
    groups_ints_test = groups_ints[test_idx]

    ############################
    ######### Fit MGGP #########
    ############################

    hier_mg_kernel = lambda params, X1, X2, groups1, groups2, group_distances: hierarchical_multigroup_kernel(
        params,
        X1,
        X2,
        groups1,
        groups2,
        group_distances,
        within_group_kernel=multigroup_rbf_covariance,
        between_group_kernel=rbf_covariance,
    )

    mggp = GP(kernel=MultiGroupRBF(), is_mggp=True)

    mggp.fit(X_train, Y_train, groups_ints_train, group_dist_mat)
    preds_mean = mggp.predict(X_test, groups_ints_test)
    curr_error = np.mean((Y_test - preds_mean) ** 2)
    errors_mggp[ii] = curr_error

    for groupnum in range(n_groups):
        # curr_idx = X_groups_test[:, groupnum] == 1
        curr_idx = groups_ints_test == groupnum
        curr_error = np.mean((Y_test[curr_idx] - preds_mean[curr_idx]) ** 2)
        errors_groupwise_mggp[ii, groupnum] = curr_error

    pbar.update(1)

    ############################
    ######### Fit HGP ##########
    ############################
    kernel = HGPKernel(within_group_kernel=RBF(), between_group_kernel=RBF())
    hgp = GP(kernel=kernel, is_hgp=True)
    hgp.fit(X_train, Y_train, groups_ints_train)
    preds_mean = hgp.predict(X_test, groups_ints_test)
    curr_error = np.mean((Y_test - preds_mean) ** 2)
    errors_hgp[ii] = curr_error

    for groupnum in range(n_groups):
        # curr_idx = groups_ints_test[:, groupnum] == 1
        curr_idx = groups_ints_test == groupnum
        curr_error = np.mean((Y_test[curr_idx] - preds_mean[curr_idx]) ** 2)
        errors_groupwise_hgp[ii, groupnum] = curr_error

    pbar.update(1)

pbar.close()

results_df = pd.melt(
    pd.DataFrame(
        {
            "MGGP": errors_mggp,
            "HGP": errors_hgp,
        }
    )
)


plt.figure(figsize=(14, 5))
plt.subplot(121)
sns.boxplot(data=results_df, x="variable", y="value")
plt.title("Total error")
plt.xlabel("")
plt.ylabel("Test MSE")
plt.tight_layout()

results_groupwise_mggp_df = pd.melt(
    pd.DataFrame(errors_groupwise_mggp, columns=tissue_labels)
)
results_groupwise_mggp_df["model"] = ["MGGP"] * results_groupwise_mggp_df.shape[0]
results_groupwise_hgp_df = pd.melt(
    pd.DataFrame(errors_groupwise_hgp, columns=tissue_labels)
)
results_groupwise_hgp_df["model"] = ["HGP"] * results_groupwise_hgp_df.shape[0]

results_df_groupwise = pd.concat(
    [
        results_groupwise_mggp_df,
        results_groupwise_hgp_df,
    ],
    axis=0,
)

plt.subplot(122)
g = sns.boxplot(data=results_df_groupwise, x="variable", y="value", hue="model")
plt.title("Group-wise error")
plt.xlabel("")
plt.ylabel("Test MSE")
g.legend_.set_title(None)
plt.tight_layout()
plt.savefig("../../plots/gtex_mggp_vs_hggp.png")
plt.show()
import ipdb

ipdb.set_trace()
