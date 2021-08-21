import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
from sklearn.linear_model import LinearRegression

DATA_DIR = "/Users/andrewjones/Documents/beehive/rrr/PRRR/data"
# DATA_FILE1 = "gtex_expression_Artery - Tibial.csv"
# DATA_FILE2 = "gtex_expression_Artery - Coronary.csv"
# DATA_FILE3 = "gtex_expression_Breast - Mammary Tissue.csv"

DATA_FILES = [
	"gtex_expression_Artery - Tibial.csv",
	"gtex_expression_Artery - Coronary.csv",
	"gtex_expression_Breast - Mammary Tissue.csv",
	"gtex_expression_Brain - Frontal Cortex (BA9).csv",
	"gtex_expression_Brain - Anterior cingulate cortex (BA24).csv",
	"gtex_expression_Brain - Cortex.csv",
	"gtex_expression_Uterus.csv",
	"gtex_expression_Vagina.csv"
]

TISSUE_NAMES = [
	"tibial_artery",
	"coronary_artery",
	"breast_mammary",
	"frontal_cortex",
	"anterior_cingulate_cortex",
	"cortex",
	"uterus",
	"vagina"
]

METADATA_FILE = "GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
metadata = pd.read_table(pjoin(DATA_DIR, METADATA_FILE))
metadata_IT = metadata[["SUBJID", "TRISCHD", "AGE"]]
metadata_IT = metadata_IT.set_index("SUBJID")

N_GENES = 100

for ii, DATA_FILE in enumerate(DATA_FILES):
	data = pd.read_csv(pjoin(DATA_DIR, DATA_FILE), index_col=0)

	gene_idx_sorted_by_variance = np.argsort(-data.var(0).values)
	if ii == 0:
		high_variance_genes = data.columns.values[gene_idx_sorted_by_variance[:N_GENES]]
	else:
		high_variance_genes = np.intersect1d(high_variance_genes, data.columns.values)

	data_high_variance = data[high_variance_genes]

	## Get subject ID and drop duplicate subjects
	data_high_variance["SUBJID"] = data_high_variance.index.str.split("-").str[:2].str.join("-")

	data_high_variance = data_high_variance.drop_duplicates(subset="SUBJID")

	data_high_variance = data_high_variance.set_index("SUBJID")

	data_subject_ids = data_high_variance.index.values
	metadata_IT_data = metadata_IT.transpose()[data_subject_ids].transpose()

	# import ipdb; ipdb.set_trace()

	data_high_variance.to_csv("../data/gtex/{}_expression.csv".format(TISSUE_NAMES[ii]))
	metadata_IT_data.to_csv("../data/gtex/{}_ischemic_time.csv".format(TISSUE_NAMES[ii]))

	assert np.array_equal(data_high_variance.index.values, metadata_IT_data.index.values)

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
	# import ipdb; ipdb.set_trace()

import ipdb; ipdb.set_trace()


# data1 = pd.read_csv(pjoin(DATA_DIR, DATA_FILE1), index_col=0)
# data2 = pd.read_csv(pjoin(DATA_DIR, DATA_FILE2), index_col=0)
# data3 = pd.read_csv(pjoin(DATA_DIR, DATA_FILE3), index_col=0)
# metadata = pd.read_table(pjoin(DATA_DIR, METADATA_FILE))

# gene_idx_sorted_by_variance = np.argsort(-data1.var(0).values)
# high_variance_genes = data1.columns.values[gene_idx_sorted_by_variance[:N_GENES]]
# high_variance_genes = np.intersect1d(high_variance_genes, data2.columns.values)
# high_variance_genes = np.intersect1d(high_variance_genes, data3.columns.values)

# data1_high_variance = data1[high_variance_genes]
# data2_high_variance = data2[high_variance_genes]
# data3_high_variance = data3[high_variance_genes]

# ## Get subject ID and drop duplicate subjects
# data1_high_variance["SUBJID"] = data1_high_variance.index.str.split("-").str[:2].str.join("-")
# data2_high_variance["SUBJID"] = data2_high_variance.index.str.split("-").str[:2].str.join("-")
# data3_high_variance["SUBJID"] = data3_high_variance.index.str.split("-").str[:2].str.join("-")

# data1_high_variance = data1_high_variance.drop_duplicates(subset="SUBJID")
# data2_high_variance = data2_high_variance.drop_duplicates(subset="SUBJID")
# data3_high_variance = data3_high_variance.drop_duplicates(subset="SUBJID")

# data1_high_variance = data1_high_variance.set_index("SUBJID")
# data2_high_variance = data2_high_variance.set_index("SUBJID")
# data3_high_variance = data3_high_variance.set_index("SUBJID")

# metadata_IT = metadata[["SUBJID", "TRISCHD"]]
# metadata_IT = metadata_IT.set_index("SUBJID")

# data1_subject_ids = data1_high_variance.index.values
# metadata_IT_data1 = metadata_IT.transpose()[data1_subject_ids].transpose()
# data2_subject_ids = data2_high_variance.index.values
# metadata_IT_data2 = metadata_IT.transpose()[data2_subject_ids].transpose()
# data3_subject_ids = data3_high_variance.index.values
# metadata_IT_data3 = metadata_IT.transpose()[data3_subject_ids].transpose()

# data1_high_variance.to_csv("../data/gtex/tibial_artery_expression.csv")
# data2_high_variance.to_csv("../data/gtex/coronary_artery_expression.csv")
# data3_high_variance.to_csv("../data/gtex/breast_mammary_expression.csv")

# metadata_IT_data1.to_csv("../data/gtex/tibial_artery_ischemic_time.csv")
# metadata_IT_data2.to_csv("../data/gtex/coronary_artery_ischemic_time.csv")
# metadata_IT_data3.to_csv("../data/gtex/breast_mammary_ischemic_time.csv")

# assert np.array_equal(data1_high_variance.index.values, metadata_IT_data1.index.values)
# assert np.array_equal(data2_high_variance.index.values, metadata_IT_data2.index.values)
# assert np.array_equal(data3_high_variance.index.values, metadata_IT_data3.index.values)

