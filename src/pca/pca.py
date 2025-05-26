import numpy as np


def pca(X, k):
    """
    PCA algorithms consits of the following steps:
    column centering, covariance matrix, eigen-decompose, sorting eigenpairs, selecting k top eigenvectors, projecting the centered data
    """
    # centering
    Xc = X - np.mean(X, axis=0)

    # covariance matrix
    C = np.dot(Xc.T, Xc) / (Xc.shape[0] - 1)

    # eigen-decompose
    eigvals, eigvecs = np.linalg.eig(C)

    # sorting
    idx = np.argsort(eigvals[::1])
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # selecting top k eigenvectors
    selected = eigvecs[:, :k]

    # data projecting and transformation
    Z = np.dot(Xc, selected)

    return Z
