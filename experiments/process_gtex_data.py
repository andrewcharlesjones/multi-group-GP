import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin

DATA_DIR = "/Users/andrewjones/Documents/beehive/rrr/PRRR/data"
DATA_FILE1 = "gtex_expression_Artery - Tibial.csv"
DATA_FILE2 = "gtex_expression_Artery - Coronary.csv"
DATA_FILE3 = "gtex_expression_Breast - Mammary Tissue.csv"

METADATA_FILE = "GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"

N_GENES = 100
data1 = pd.read_csv(pjoin(DATA_DIR, DATA_FILE1), index_col=0)
data2 = pd.read_csv(pjoin(DATA_DIR, DATA_FILE2), index_col=0)
data3 = pd.read_csv(pjoin(DATA_DIR, DATA_FILE3), index_col=0)
metadata = pd.read_table(pjoin(DATA_DIR, METADATA_FILE))

gene_idx_sorted_by_variance = np.argsort(-data1.var(0).values)
high_variance_genes = data1.columns.values[gene_idx_sorted_by_variance[:N_GENES]]
high_variance_genes = np.intersect1d(high_variance_genes, data2.columns.values)
high_variance_genes = np.intersect1d(high_variance_genes, data3.columns.values)

data1_high_variance = data1[high_variance_genes]
data2_high_variance = data2[high_variance_genes]
data3_high_variance = data3[high_variance_genes]

## Get subject ID and drop duplicate subjects
data1_high_variance["SUBJID"] = data1_high_variance.index.str.split("-").str[:2].str.join("-")
data2_high_variance["SUBJID"] = data2_high_variance.index.str.split("-").str[:2].str.join("-")
data3_high_variance["SUBJID"] = data3_high_variance.index.str.split("-").str[:2].str.join("-")

data1_high_variance = data1_high_variance.drop_duplicates(subset="SUBJID")
data2_high_variance = data2_high_variance.drop_duplicates(subset="SUBJID")
data3_high_variance = data3_high_variance.drop_duplicates(subset="SUBJID")

data1_high_variance = data1_high_variance.set_index("SUBJID")
data2_high_variance = data2_high_variance.set_index("SUBJID")
data3_high_variance = data3_high_variance.set_index("SUBJID")

metadata_IT = metadata[["SUBJID", "TRISCHD"]]
metadata_IT = metadata_IT.set_index("SUBJID")

data1_subject_ids = data1_high_variance.index.values
metadata_IT_data1 = metadata_IT.transpose()[data1_subject_ids].transpose()
data2_subject_ids = data2_high_variance.index.values
metadata_IT_data2 = metadata_IT.transpose()[data2_subject_ids].transpose()
data3_subject_ids = data3_high_variance.index.values
metadata_IT_data3 = metadata_IT.transpose()[data3_subject_ids].transpose()

data1_high_variance.to_csv("../data/gtex/tibial_artery_expression.csv")
data2_high_variance.to_csv("../data/gtex/coronary_artery_expression.csv")
data3_high_variance.to_csv("../data/gtex/breast_mammary_expression.csv")

metadata_IT_data1.to_csv("../data/gtex/tibial_artery_ischemic_time.csv")
metadata_IT_data2.to_csv("../data/gtex/coronary_artery_ischemic_time.csv")
metadata_IT_data3.to_csv("../data/gtex/breast_mammary_ischemic_time.csv")

assert np.array_equal(data1_high_variance.index.values, metadata_IT_data1.index.values)
assert np.array_equal(data2_high_variance.index.values, metadata_IT_data2.index.values)
assert np.array_equal(data3_high_variance.index.values, metadata_IT_data3.index.values)
import ipdb; ipdb.set_trace()


