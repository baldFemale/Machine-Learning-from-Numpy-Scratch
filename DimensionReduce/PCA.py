import numpy as np


class PCA:

    def __init__(self):
        self.w = None
        pass

    def fit(self, X, n_component):
        X = X.T
        mean = np.mean(X,axis=0)
        X = X-mean
        eigen_value,eigen_vector = np.linalg.eig(X @ X.T)
        index = np.argsort(eigen_value)[::-1]
        self.w = eigen_vector[index[:n_component]]

    def transform(self, X):
        X = X.T
        return (self.w @ X).T
