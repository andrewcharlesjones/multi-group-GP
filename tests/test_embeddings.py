from multigroupGP import multigroup_rbf_kernel, embed_distance_matrix
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import sqrtm
from sklearn.manifold import MDS

def test_embeddings():

	n_groups = 3
	n = 100

	for _ in range(10):
		X_embeddings = np.random.normal(size=(n_groups, n_groups))

		group_distances = pairwise_distances(X_embeddings)

		embedding = embed_distance_matrix(group_distances)

		embedding_dists = pairwise_distances(embedding)

		assert np.allclose(embedding_dists, group_distances)


if __name__ == "__main__":
	test_embeddings()
