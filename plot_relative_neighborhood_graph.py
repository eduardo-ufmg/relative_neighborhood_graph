import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plot_relative_neighborhood_graph(
    X: ArrayLike, adjacency_matrix: np.ndarray
) -> None:
    """
    Plot the relative neighborhood graph of a bi-dimensional set of points.

    Parameters
    ----------
    X : ArrayLike
        An array of shape (n_samples, n_features) representing the points.
    adjacency_matrix : np.ndarray
        An adjacency matrix representing the relative neighborhood graph.
    """
    X = np.asarray(X)

    if X.shape[1] != 2:
        raise ValueError("The input points must be bi-dimensional (2D).")

    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1])

    n_samples = X.shape[0]

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if adjacency_matrix[i, j]:
                plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]])

    plt.title("Relative Neighborhood Graph")
    plt.show()
