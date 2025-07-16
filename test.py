import numpy as np
from sklearn.datasets import make_blobs
from relative_neighborhood_graph import relative_neighborhood_graph
from plot_relative_neighborhood_graph import plot_relative_neighborhood_graph

# Generate synthetic data using sklearn's make_blobs
X, _ = make_blobs(n_samples=100, n_features=2)[0:2]

# Compute the relative neighborhood graph
adjacency_matrix = relative_neighborhood_graph(X)

# Plot the relative neighborhood graph
plot_relative_neighborhood_graph(X, adjacency_matrix)