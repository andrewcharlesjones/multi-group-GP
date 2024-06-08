import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
from sklearn.linear_model import LinearRegression
import os
from tqdm import tqdm


DATA_DIR = "/Users/andrewjones/Documents/beehive/rrr/PRRR/data/gtex"
SAVE_DIR = "../../data/gtex"
# DATA_FILE1 = "gtex_expression_Artery - Tibial.csv"
# DATA_FILE2 = "gtex_expression_Artery - Coronary.csv"
# DATA_FILE3 = "gtex_expression_Breast - Mammary Tissue.csv"


datafile_prefix = "gtex_expression_"
DATA_FILES = [x for x in os.listdir(DATA_DIR) if x.startswith(datafile_prefix)]
# import ipdb; ipdb.set_trace()

TISSUE_NAMES = [
    x[len(datafile_prefix) : -4].replace(" - ", "_").replace(" ", "_")
    for x in DATA_FILES
]

## Get all shared genes between tissue types
# all_shared_genes = pd.read_csv(pjoin(DATA_DIR, DATA_FILES[0]), index_col=0, nrows=0).columns
# import ipdb; ipdb.set_trace()
# for jj in np.arange(1, 3): #len(TISSUE_NAMES)):
#     curr_genes = pd.read_csv(pjoin(DATA_DIR, DATA_FILES[jj]), index_col=0, nrows=0).columns.tolist()
#     all_shared_genes = np.intersect1d(curr_genes, all_shared_genes)


# DATA_FILES = [
#     "gtex_expression_Artery - Tibial.csv",
#     "gtex_expression_Artery - Coronary.csv",
#     "gtex_expression_Breast - Mammary Tissue.csv",
#     "gtex_expression_Brain - Frontal Cortex (BA9).csv",
#     "gtex_expression_Brain - Anterior cingulate cortex (BA24).csv",
#     "gtex_expression_Brain - Cortex.csv",
#     "gtex_expression_Uterus.csv",
#     "gtex_expression_Vagina.csv",
# ]

# TISSUE_NAMES = [
#     "tibial_artery",
#     "coronary_artery",
#     "breast_mammary",
#     "frontal_cortex",
#     "anterior_cingulate_cortex",
#     "cortex",
#     "uterus",
#     "vagina",
# ]

METADATA_FILE = "GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
metadata = pd.read_table(pjoin(DATA_DIR, METADATA_FILE))
metadata_IT = metadata[["SUBJID", "TRISCHD", "AGE"]]
metadata_IT = metadata_IT.set_index("SUBJID")

N_GENES = 1000

# missing_tissues = ['Brain_Anterior_cingulate_cortex_(BA24)_expression.csv',
#  'Brain_Caudate_(basal_ganglia)_expression.csv',
#  'Brain_Cerebellum_expression.csv',
#  'Brain_Frontal_Cortex_(BA9)_expression.csv',
#  'Brain_Hippocampus_expression.csv', 'Colon_Transverse_expression.csv']

for ii in tqdm(range(len(DATA_FILES))):

    curr_gene_exp_fname = pjoin(SAVE_DIR, "{}_expression.csv".format(TISSUE_NAMES[ii]))
    # if os.path.isfile(curr_gene_exp_fname) and ii != 0:
    #     continue

    # if ii != 0 and "{}_expression.csv".format(TISSUE_NAMES[ii]) not in missing_tissues:
    #     continue

    print("Saving {}".format(TISSUE_NAMES[ii]))

    DATA_FILE = DATA_FILES[ii]
    if DATA_FILE == "gtex_expression_Muscle - Skeletal.csv":
        continue

    data = pd.read_csv(pjoin(DATA_DIR, DATA_FILE), index_col=0)

    if ii == 0:

        gene_idx_sorted_by_variance = np.argsort(-data.var(0).values)
        high_variance_genes = data.columns.values[gene_idx_sorted_by_variance[:N_GENES]]
    # else:
    #     high_variance_genes = np.intersect1d(high_variance_genes, data.columns.values)

    data_high_variance = data[high_variance_genes]

    ## Get subject ID and drop duplicate subjects
    data_high_variance["SUBJID"] = (
        data_high_variance.index.str.split("-").str[:2].str.join("-")
    )

    data_high_variance = data_high_variance.drop_duplicates(subset="SUBJID")

    data_high_variance = data_high_variance.set_index("SUBJID")

    data_subject_ids = data_high_variance.index.values
    metadata_IT_data = metadata_IT.transpose()[data_subject_ids].transpose()

    assert np.array_equal(
        data_high_variance.index.values, metadata_IT_data.index.values
    )
    # import ipdb; ipdb.set_trace()

    assert data_high_variance.shape[1] == N_GENES
    assert np.array_equal(data_high_variance.columns.values, high_variance_genes)

    data_high_variance.to_csv(curr_gene_exp_fname)
    metadata_IT_data.to_csv(
        pjoin(SAVE_DIR, "{}_ischemic_time.csv".format(TISSUE_NAMES[ii]))
    )

    # reg_data = np.log(data_high_variance.values + 1)
    # reg_data = (reg_data - reg_data.mean(0)) / reg_data.std(0)
    # reg = LinearRegression()

    # regX = reg_data
    # regY = metadata_IT_data.dropna(axis=1).convert_dtypes("float64")._get_numeric_data()

    # scores = np.empty(regY.shape[1])
    # for jj in range(regY.shape[1]):
    # 	curr_output = regY.values[:, jj]
    # 	reg.fit(regX, curr_output)
    # 	score = reg.score(regX, curr_output)
    # 	scores[jj] = score
    # 	print("{}: {}".format(regY.columns.values[jj], score))
    import ipdb; ipdb.set_trace()

import ipdb

ipdb.set_trace()
