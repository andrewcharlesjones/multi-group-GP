import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import stan
import arviz as az
from scipy.stats import pearsonr, spearmanr

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

import socket

if socket.gethostname() == "andyjones":
    MODEL_DIR = "../../multigroupGP/models"
else:
    MODEL_DIR = "../../models"

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

# DATA_DIR = "../../data/gtex"
DATA_DIR = "../../data/gtex/logtpm"


data_prefixes = [
    "Brain_Amygdala",
    "Brain_Anterior_cingulate_cortex_(BA24)",
    "Brain_Caudate_(basal_ganglia)",
    # "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum",
    "Brain_Cortex",
    # "Brain_Frontal_Cortex_(BA9)",
    "Brain_Hippocampus",
    "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_(basal_ganglia)",
    "Brain_Putamen_(basal_ganglia)",
    "Brain_Spinal_cord_(cervical_c-1)",
    "Brain_Substantia_nigra",
]

tissue_labels = [" ".join(x.split("_")[1:]) for x in data_prefixes]


data_fnames = [x + "_expression.csv" for x in data_prefixes]
# output_fnames = [x + "_ischemic_time.csv" for x in data_prefixes]
output_fnames = [x + "_metadata.csv" for x in data_prefixes]

# output_col = "TRISCHD"
output_col = "SMTSISCH"

N_MCMC_ITER = 200
N_WARMUP_ITER = 50
N_CHAINS = 4

PCA_REDUCE = False
n_components = 5

n_samples = None

n_groups = len(tissue_labels)

group_dist_mat = 1 - np.eye(len(tissue_labels))


X_list = []
Y_list = []
groups_list = []
groups_ints = []
groups_oh = []
for kk in range(n_groups):
    data = pd.read_csv(pjoin(DATA_DIR, data_fnames[kk]), index_col=0)
    output = pd.read_csv(pjoin(DATA_DIR, output_fnames[kk]), index_col=0)[output_col]
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
    curr_groups_oh = np.repeat([curr_group_one_hot], curr_n, axis=0)
    groups_oh.append(curr_groups_oh)

    X_list.append(curr_X)
    Y_list.append(curr_Y)
    # groups_list.append(curr_groups)

    groups_ints.append(np.repeat(kk, curr_X.shape[0]))
    gene_names = data.columns.values

groups = np.concatenate(groups_ints)
groups_oh = np.concatenate(groups_oh)

X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list).squeeze()
ntotal = X.shape[0]

X = X[:, X.std(0) > 0]


# X = np.log(X + 1)
X = (X - X.mean(0)) / X.std(0)

# Y = np.log(Y + 1)
Y = (Y - Y.mean(0)) / Y.std(0)

if PCA_REDUCE:
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)


### Posterior inference ###


gene_corrs_with_it = np.abs(
    np.array([spearmanr(X[:, ii], Y)[0] for ii in range(X.shape[1])])
)
highest_corr_idx = np.argmax(gene_corrs_with_it)
X = X[:, highest_corr_idx]
gene_name = gene_names[highest_corr_idx]


data = {
    "P": 1,
    "ngroups": n_groups,
    "N": X.shape[0],
    "x": X.reshape(-1, 1),
    "design": groups_oh,
    "y": Y,
    "groups": groups.astype(int),
    "group_dist_mat": group_dist_mat,
}



## Load model
with open(pjoin(MODEL_DIR, "mggp_collapsed.stan"), "r") as file:
    model_code = file.read()


# Set up model
posterior = stan.build(model_code, data=data)

# Start sampling
fit = posterior.sample(
    num_chains=N_CHAINS, num_warmup=N_WARMUP_ITER, num_samples=N_MCMC_ITER
)

arviz_summary = az.summary(fit)

pd.DataFrame(X).to_csv("./out/stan_out/X_train.csv")
pd.DataFrame(Y).to_csv("./out/stan_out/Y_train.csv")
pd.DataFrame(groups).to_csv("./out/stan_out/groups_train.csv")

cov_params_df = pd.DataFrame(
    {
        "outputvariance": fit["outputvariance"].squeeze(),
        "lengthscale": fit["lengthscale"].squeeze(),
        "alpha": fit["alpha"].squeeze(),
    }
)
noise_variance_df = pd.DataFrame(fit["sigma"].T)
beta_df = pd.DataFrame(fit["beta"])

cov_params_df.to_csv("./out/stan_out/cov_params_samples.csv")
noise_variance_df.to_csv("./out/stan_out/noise_variance_samples.csv")
beta_df.to_csv("./out/stan_out/beta_samples.csv")
import ipdb; ipdb.set_trace()

