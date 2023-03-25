import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# Generate some random data for clustering
X, y = make_blobs(n_samples=1000, centers=3, random_state=42)

# Create and fit the HDBSCAN model
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5)
hdbscan_model.fit(X)

# Get the cluster labels (-1 indicates an outlier)
labels = hdbscan_model.labels_

# Plot the data points colored by their cluster label
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()