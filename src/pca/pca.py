import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


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
    idx = np.argsort(eigvals[::-1])
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # selecting top k eigenvectors
    selected = eigvecs[:, :k]

    # data projecting and transformation
    Z = np.dot(Xc, selected)

    return Z

data = load_iris()
X, y = data.data, data.target

Z = pca(X, k=2)

# before and after plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original data: Sepal Length vs Sepal Width
axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
axes[0].set_xlabel(data.feature_names[0])
axes[0].set_ylabel(data.feature_names[1])
axes[0].set_title("Original: Sepal Length vs Sepal Width")
axes[0].grid(True)

# PCA-transformed data: PC1 vs PC2
axes[1].scatter(Z[:, 0], Z[:, 1], c=y, edgecolor='k')
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].set_title("PCA-Transformed Data")
axes[1].grid(True)

plt.tight_layout()
plt.show()
