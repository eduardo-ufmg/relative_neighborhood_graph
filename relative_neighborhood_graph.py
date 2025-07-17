import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay


def relative_neighborhood_graph(X: ArrayLike) -> np.ndarray:
    """
    Compute the relative neighborhood graph of a set of points.

    Parameters
    ----------
    X : ArrayLike
        An array of shape (n_samples, n_features) representing the points.

    Returns
    -------
    np.ndarray
        An adjacency matrix representing the relative neighborhood graph.
    """
    X = np.asarray(X)
    n_samples = X.shape[0]

    if n_samples < 2:
        return np.zeros((n_samples, n_samples), dtype=bool)

    # Compute the Delaunay triangulation
    delaunay = Delaunay(X)

    # Initialize adjacency matrix
    adjacency_matrix = np.zeros((n_samples, n_samples), dtype=bool)

    # Iterate over each simplex in the Delaunay triangulation
    for simplex in delaunay.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                p1, p2 = simplex[i], simplex[j]
                distance = np.linalg.norm(X[p1] - X[p2])

                # Check if there is a third point that is closer to both p1 and p2
                for k in range(n_samples):
                    if k not in simplex:
                        d1 = np.linalg.norm(X[p1] - X[k])
                        d2 = np.linalg.norm(X[p2] - X[k])
                        if d1 < distance and d2 < distance:
                            break
                else:
                    # If no such point exists, add edge to adjacency matrix
                    adjacency_matrix[p1, p2] = True
                    adjacency_matrix[p2, p1] = True

    return adjacency_matrix.astype(int)
