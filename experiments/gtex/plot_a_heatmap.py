import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc

import matplotlib

font = {"size": 15}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

plt.figure(figsize=(7, 7))
a_mat = pd.read_csv("./a_matrix_full.csv", index_col=0)
a_mat_log = np.log(a_mat.values)
np.fill_diagonal(a_mat_log, None)
a_mat_log_df = pd.DataFrame(a_mat_log, columns=a_mat.columns, index=a_mat.index)
# a_mat = np.log(a_mat)
# np.fill_diagonal(a_mat, 0)
# import ipdb; ipdb.set_trace()
# linkage = hc.linkage(sp.distance.squareform(a_mat_log_df - np.min(a_mat_log_df.values)), method='average')
#
sns.clustermap(a_mat)  # , row_linkage=linkage, col_linkage=linkage)
plt.title(r"Pairwise estimated $\log(a)$")
plt.tight_layout()
plt.show()

import ipdb

ipdb.set_trace()
