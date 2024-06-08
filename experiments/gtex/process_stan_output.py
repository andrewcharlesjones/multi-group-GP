import matplotlib
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pcpca import PCPCA
import numpy as np
import stan
from hashlib import md5
from os.path import join as pjoin
import pickle
import os
from multigroupGP import MultiGroupRBF
import arviz as az
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.linalg import cho_solve

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


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
n_groups = len(tissue_labels)


X = pd.read_csv("./out/stan_out/X_train.csv", index_col=0).values
Y = pd.read_csv("./out/stan_out/Y_train.csv", index_col=0).values
groups = pd.read_csv(
    "./out/stan_out/groups_train.csv", index_col=0
).values.squeeze()

cov_params_df = pd.read_csv("./out/stan_out/cov_params_samples.csv", index_col=0)
noise_variance_df = pd.read_csv("./out/stan_out/noise_variance_samples.csv", index_col=0)
beta_df = pd.read_csv("./out/stan_out/beta_samples.csv", index_col=0)

n = X.shape[0]


percentiles = [2.5, 50, 97.5]


cov_params_percentile_df = pd.DataFrame(
    np.percentile(cov_params_df, q=percentiles, axis=0),
    columns=cov_params_df.columns,
)
cov_params_percentile_df.columns = ["$\\sigma^2$", "$b$", "$a$"]
beta_percentiles_df = pd.DataFrame({
    "$\\beta_{" + str(ii + 1) + "}$": np.percentile(beta_df.iloc[ii, :], q=percentiles) for ii in range(beta_df.shape[0])
})
noise_variance_percentiles_df = pd.DataFrame({
    "$\\tau^2_{" + str(ii + 1) + "}$": np.percentile(noise_variance_df.iloc[:, ii], q=percentiles) for ii in range(noise_variance_df.shape[1])
})

percentiles_df = pd.concat([
    cov_params_percentile_df,
    beta_percentiles_df,
    noise_variance_percentiles_df
], axis=1)


table_df = pd.DataFrame(np.zeros((percentiles_df.shape[1], 2)))
table_df.iloc[:, 0] = percentiles_df.columns.values
for ii in range(percentiles_df.shape[1]):
    table_df.iloc[ii, 1] = (
        str(round(percentiles_df.iloc[1, ii], 2))
        + " ("
        + str(round(percentiles_df.iloc[0, ii], 2))
        + "; "
        + str(round(percentiles_df.iloc[2, ii], 2))
        + ")"
    )

# table_df.index = percentiles_df.columns
table_df.columns = ["\\textbf{Parameter}", "\\textbf{Posterior percentiles}"]

with open('./out/stan_out/posterior_percentiles_table_gtex.txt', 'w') as ff:
    
    ff.write(table_df.columns.values[0] + " & " + table_df.columns.values[1] + " \\\ \n")
    ff.write("\\hline\n")

    for ii in range(len(table_df)):
            
        ff.write(str(table_df.values[ii, 0]) + " & " + str(table_df.values[ii, 1]) + " \\\ \n")
        ff.write("\\hline\n")
    # the_file.write('Hello\n')
# table_df.values[ii, 1]
# table_df.to_csv("./out/stan_out/posterior_percentiles_table_gtex.csv", index=False, sep=",")

# import ipdb; ipdb.set_trace()



def multigroup_RBF(x, groups, sample_group_dist_mat, lengthscale, outputvariance, alpha):
    squared_dist_mat = ((x - x.T) ** 2)
    # group_diff_scaling = group_dist_mat[groups[i] + 1, groups[j] + 1]
    scaling_term = 1 / (sample_group_dist_mat * alpha ** 2 + 1) ** 0.5
    K = scaling_term * np.exp(-0.5 * squared_dist_mat * lengthscale / (sample_group_dist_mat * alpha ** 2 + 1));
    K += 1e-8 * np.eye(n)
    return K

# import ipdb; ipdb.set_trace()
group_dist_mat = 1 - np.eye(n_groups)
sample_group_dist_mat = np.zeros((n, n))
for ii in range(n):
    for jj in range(ii, n):
        sample_group_dist_mat[ii, jj] = group_dist_mat[groups[ii], groups[jj]]
        sample_group_dist_mat[jj, ii] = sample_group_dist_mat[ii, jj]


n_mcmc_samples = len(cov_params_df)


## Sample latent process
n_samples = n_mcmc_samples
f_samples = np.zeros((n_samples, n))
y_samples = np.zeros((n_samples, n))
for ii in tqdm(range(n_samples)):

    curr_outputvariance, curr_lengthscale, curr_alpha = cov_params_df.iloc[ii].values

    curr_sigma = noise_variance_df.values.squeeze()[ii]
    curr_beta = beta_df.values[:, ii]
    curr_K = multigroup_RBF(X, groups, sample_group_dist_mat, outputvariance=curr_outputvariance, lengthscale=curr_lengthscale, alpha=curr_alpha)
    

    curr_K_noisy = curr_K + np.diag(curr_sigma[groups])
    K_chol = np.linalg.cholesky(curr_K_noisy)
    K_noisy_inv = cho_solve((K_chol, True), np.eye(n))
        

    curr_mean = curr_beta[groups]
    curr_m = Y.squeeze() - curr_mean
    curr_premult_K = curr_K @ K_noisy_inv

    f_samples[ii] = mvn.rvs(mean=curr_premult_K @ curr_m, cov=curr_premult_K  @ curr_K)
    y_samples[ii] = f_samples[ii] + curr_mean


f_samples_mean = f_samples.mean(0)
y_samples_mean = y_samples.mean(0)
y_samples_stddev = y_samples.std(0)
# means_mean = fit["means"].mean(1)

# colors = ["blue", "red"]
# for ii in range(2):
#     curr_idx = np.where(groups == ii)
#     plt.scatter(X[curr_idx], Y[curr_idx], color=colors[ii], alpha=0.3)
#     plt.scatter(X[curr_idx], y_samples_mean[curr_idx], color=colors[ii])
# plt.show()


# import ipdb; ipdb.set_trace()

plt.figure(figsize=(30, 24))
n_rows = 4
n_cols = 3
for gg_main in range(n_groups):

    plt.subplot(n_rows, n_cols, gg_main + 1)
    
    for gg in range(n_groups):
        curr_idx = np.where(groups == gg)[0]
        plt.scatter(X[curr_idx], Y[curr_idx], color="gray")

    curr_idx_main = np.where(groups == gg_main)[0]
    curr_sorted_idx = np.argsort(X[curr_idx_main].squeeze())

    plt.fill_between(
        X[curr_idx_main][curr_sorted_idx].squeeze(),
        y_samples_mean[curr_idx_main][curr_sorted_idx] - 2 * y_samples_stddev[curr_idx_main][curr_sorted_idx],
        y_samples_mean[curr_idx_main][curr_sorted_idx] + 2 * y_samples_stddev[curr_idx_main][curr_sorted_idx],
        alpha=0.2,
        color="red",
    )

    
    plt.plot(
        X[curr_idx_main][curr_sorted_idx],
        y_samples_mean[curr_idx_main][curr_sorted_idx],
        color="black",
        linestyle="--",
        linewidth=4,
    )

    plt.scatter(X[curr_idx_main], Y[curr_idx_main], color="red")

    plt.title(tissue_labels[gg_main])

    plt.xlabel(r"$\emph{TXNIP}$ expression")
    plt.ylabel("Ischemic time")
    plt.tight_layout()
# plt.savefig(
#     "./out/group_specific_posterior_plots/gtex_posterior_inference_all_groups_subplots.png".format(
#         tissue_labels[gg_main].replace(" ", "_"), dpi=300
#     )
# )
plt.show()
plt.close()
# import ipdb; ipdb.set_trace()

plt.figure(figsize=(17, 6))
for gg in range(n_groups):
    curr_idx_main = np.where(groups == gg)[0]
    curr_sorted_idx = np.argsort(X[curr_idx_main].squeeze())


    
    plt.plot(
        X[curr_idx_main][curr_sorted_idx],
        y_samples_mean[curr_idx_main][curr_sorted_idx],
        # color="red",
        # linestyle="--",
        # linewidth=4,
        label=tissue_labels[gg],
    )


    plt.scatter(X[curr_idx_main], Y[curr_idx_main])

plt.xlabel(r"$\emph{TXNIP}$ expression")
plt.ylabel("Ischemic time")
plt.legend()
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=20)
plt.tight_layout()
# plt.savefig("./out/group_specific_posterior_plots/gtex_posterior_inference_all_groups.png", dpi=300)
plt.show()


import ipdb

ipdb.set_trace()
