import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
from sklearn.linear_model import LinearRegression
import os
from tqdm import tqdm


METADATA_FILE = "/Users/andrewjones/Documents/beehive/rrr/PRRR/data/gtex/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
DATA_FILE = "/Users/andrewjones/Documents/beehive/multi-group-GP/data/gtex/gtex_expression_logtpm_top5000genes.csv"
SAVE_DIR = "../../data/gtex/logtpm"
ISCHEMIC_TIME_COLNAME = "SMTSISCH"
TISSUE_COLNAME = "SMTSD"

data = pd.read_csv(DATA_FILE, index_col=0)
metadata = pd.read_table(METADATA_FILE)
metadata.index = metadata["SAMPID"].values

assert len(data.index.values) == len(np.unique(data.index.values))
assert len(metadata["SAMPID"].values) == len(np.unique(metadata["SAMPID"].values))

shared_sample_ids = np.intersect1d(data.index.values, metadata["SAMPID"].values)
assert len(shared_sample_ids) == data.shape[0]

## Align datasets
data = data.transpose()[shared_sample_ids].transpose()
metadata = metadata.transpose()[shared_sample_ids].transpose()

assert np.array_equal(data.index.values, metadata.index.values)

tissue_counts = metadata.SMTSD.value_counts()
tissues_to_keep = tissue_counts.index.values[tissue_counts > 100]

tissues_to_keep_idx = np.where(
    metadata[TISSUE_COLNAME].isin(tissues_to_keep).values == True
)[0]

data = data.iloc[tissues_to_keep_idx, :]
metadata = metadata.iloc[tissues_to_keep_idx, :]

for curr_tissue in tissues_to_keep:

    tissue_string_formatted = curr_tissue.replace(" - ", "_")
    tissue_string_formatted = tissue_string_formatted.replace(" ", "_")
    print("Saving {}".format(tissue_string_formatted))

    curr_idx = np.where(metadata[TISSUE_COLNAME].values == curr_tissue)[0]
    curr_data = data.iloc[curr_idx, :]
    curr_metadata = metadata.iloc[curr_idx, :]
    assert np.array_equal(data.index.values, metadata.index.values)

    curr_data.to_csv(pjoin(SAVE_DIR, "{}_expression.csv".format(tissue_string_formatted)))
    curr_metadata.to_csv(pjoin(SAVE_DIR, "{}_metadata.csv".format(tissue_string_formatted)))

    # import ipdb; ipdb.set_trace()

import ipdb

ipdb.set_trace()
