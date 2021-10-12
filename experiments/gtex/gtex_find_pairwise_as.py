# import numpy as np
import numpy as np
import numpy.random as npr
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc
import os
import sys

# sys.path.append("../../models")
# sys.path.append("../../kernels")
# from gp import (
#     GP,
# )
# from kernels import (
#     multigroup_rbf_kernel,
# )
from multigroupGP import GP, multigroup_rbf_kernel

import matplotlib

font = {"size": 15}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

DATA_DIR = "../../data/gtex"

all_fnames = os.listdir(DATA_DIR)

data_fnames = np.sort([x for x in all_fnames if "_expression.csv" in x])
output_fnames = np.sort([x for x in all_fnames if "_ischemic_time.csv" in x])

data_sizes = [
    pd.read_csv(pjoin(DATA_DIR, data_fnames[ii])).shape
    for ii in range(len(data_fnames))
]
sample_sizes = [x[0] for x in data_sizes]
feature_sizes = [x[1] for x in data_sizes]

# print(data_fnames[np.where(np.array(feature_sizes) == 100)[0]])
assert np.all(np.array(feature_sizes) == 101)
# import ipdb

# ipdb.set_trace()


# n_tissues = 15
# rand_idx = np.random.choice(np.arange(len(data_fnames)), size=n_tissues, replace=False)
# data_fnames = data_fnames[rand_idx]
# output_fnames = output_fnames[rand_idx]


tissue_labels = [" ".join(x.split("_")[:-1]) for x in data_fnames]
tissue_labels_it = [" ".join(x.split("_")[:-2]) for x in output_fnames]

assert np.array_equal(tissue_labels, tissue_labels_it)


# tissue_labels = [
#     "Anterior\ncingulate\ncortex",
#     "Frontal\ncortex",
#     "Cortex",
#     "Uterus",
#     "Vagina",
#     "Breast",
#     "Tibial\nartery",
#     "Coronary\nartery",
# ]
# data_prefixes = [
#     "anterior_cingulate_cortex",
#     "frontal_cortex",
#     "cortex",
#     "uterus",
#     "vagina",
#     "breast_mammary",
#     "tibial_artery",
#     "coronary_artery",
# ]
# data_fnames = [x + "_expression.csv" for x in data_prefixes]
# output_fnames = [x + "_ischemic_time.csv" for x in data_prefixes]

output_col = "TRISCHD"


n_repeats = 5
n_genes = 50
n_samples = None

n_groups = len(tissue_labels)
n_groups_per_test = 2


a_matrix = np.zeros((n_repeats, n_groups, n_groups))

total_num_runs = n_repeats * (n_groups * (n_groups - 1)) / 2
pbar = tqdm(total=total_num_runs)

for ii in range(n_repeats):

    for jj in range(n_groups):

        ## First group
        X_list = []
        Y_list = []
        groups_list = []
        groups_ints = []

        data1 = pd.read_csv(pjoin(DATA_DIR, data_fnames[jj]), index_col=0)
        if data1.shape[1] != 100:
            # pbar.update(1)
            # continue
            raise Exception("Wrong number of features")

        output = pd.read_csv(pjoin(DATA_DIR, output_fnames[jj]), index_col=0)[
            output_col
        ]

        assert np.array_equal(data1.index.values, output.index.values)

        curr_X, curr_Y = data1.values, output.values

        if n_samples is None:
            rand_idx = np.arange(curr_X.shape[0])
        else:
            rand_idx = np.random.choice(
                np.arange(curr_X.shape[0]),
                replace=False,
                size=min(n_samples, curr_X.shape[0]),
            )

        curr_X_group0 = curr_X[rand_idx]
        curr_Y_group0 = curr_Y[rand_idx]

        curr_n = curr_X_group0.shape[0]

        curr_group_one_hot = np.zeros(n_groups_per_test)
        curr_group_one_hot[0] = 1
        curr_groups_group0 = np.repeat([curr_group_one_hot], curr_n, axis=0)

        # X_list.append(curr_X)
        # Y_list.append(curr_Y)
        # groups_list.append(curr_groups)

        group0_group_ints = np.repeat(0, curr_X_group0.shape[0])
        # groups_ints.append()

        for kk in range(jj):

            pbar.update(1)

            groups_ints = [group0_group_ints]
            X_list = [curr_X_group0]
            Y_list = [curr_Y_group0]
            groups_list = [curr_groups_group0]

            data2 = pd.read_csv(pjoin(DATA_DIR, data_fnames[kk]), index_col=0)
            assert np.array_equal(data1.columns.values, data2.columns.values)

            if data2.shape[1] != 100:
                raise Exception("Wrong number of features")

            output = pd.read_csv(pjoin(DATA_DIR, output_fnames[kk]), index_col=0)[
                output_col
            ]
            assert np.array_equal(data2.index.values, output.index.values)

            curr_X, curr_Y = data2.values, output.values

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

            curr_group_one_hot = np.zeros(n_groups_per_test)
            curr_group_one_hot[1] = 1
            curr_groups = np.repeat([curr_group_one_hot], curr_n, axis=0)

            X_list.append(curr_X)
            Y_list.append(curr_Y)
            groups_list.append(curr_groups)

            groups_ints.append(np.repeat(1, curr_X.shape[0]))

            X_groups = np.concatenate(groups_list, axis=0)
            groups_ints = np.concatenate(groups_ints)

            X = np.concatenate(X_list, axis=0)
            Y = np.concatenate(Y_list).squeeze()
            ntotal = X.shape[0]

            X = np.log(X + 1)
            X = (X - X.mean(0)) / X.std(0)

            # Y = np.log(Y + 1)
            Y = (Y - Y.mean(0)) / Y.std(0)

            # pca = PCA(n_components=n_genes)
            # X = pca.fit_transform(X)
            X = X[:, :n_genes]

            ############################
            ######### Fit MGGP #########
            ############################

            mggp = GP(kernel=multigroup_rbf_kernel, is_mggp=True)
            mggp.fit(X, Y, groups_ints, verbose=False, print_every=1, tol=1.)

            estimated_a = np.exp(mggp.params[-2]) + 1e-6
            a_matrix[ii, jj, kk] = estimated_a
            
pbar.close()

a_matrix_mean = np.mean(a_matrix, axis=0)
# mask = np.triu(a_matrix_mean)
mask = np.zeros_like(a_matrix_mean, dtype=bool)
mask[np.triu_indices_from(mask)] = True


a_matrix_full = np.triu(a_matrix_mean.T, 1) + a_matrix_mean
linkage = hc.linkage(sp.distance.squareform(a_matrix_full), method="average")
a_matrix_full_df = pd.DataFrame(a_matrix_full)
a_matrix_full_df.index = tissue_labels
a_matrix_full_df.columns = tissue_labels
# sns.clustermap(a_matrix_full_df, row_linkage=linkage, col_linkage=linkage)
# plt.show()


a_matrix_full_df.columns = tissue_labels
a_matrix_full_df.index = tissue_labels
a_matrix_full_df.to_csv("./a_matrix_full.csv")

# plt.figure(figsize=(10, 8))
# g = sns.heatmap(
#     a_matrix_mean,
#     xticklabels=tissue_labels,
#     yticklabels=tissue_labels,
#     annot=True,
#     mask=mask,
#     linewidths=4,
#     linecolor="gray",
# )
# g.set_yticklabels(g.get_yticklabels(), rotation=0)  # , fontsize = 8)
# g.set_xticklabels(g.get_xticklabels(), rotation=90)  # , fontsize = 8)
# plt.tight_layout()
# plt.savefig("../../plots/gtex_pairwise_a_heatmap.png")
# plt.close()

# import ipdb

# ipdb.set_trace()
