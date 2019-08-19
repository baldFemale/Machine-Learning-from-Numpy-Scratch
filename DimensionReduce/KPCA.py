import numpy as np


class KPCA:

    def __init__(self):
        self.alpha = None

    def high_dimension_map(self, x):
        """
        feel free to try other kernel functions
        """
        return x*x

    def kernal_func(self, x1, x2):
        return self.high_dimension_map(x1).T @ self.high_dimension_map(x2)

    def fit(self, X, n_component):
        X = X.T # n*m
        K = self.kernal_func(X, X)
        print(K.shape)
        eigen_value , eigen_vector = np.linalg.eig(K)
        index = np.argsort(eigen_value)[::-1]
        self.alpha = (eigen_vector[index[:n_component]]).T

    def transform(self, X):
        X = X.T # n*m
        print(self.high_dimension_map(X).shape)
        print(self.alpha.shape)
        print(X.shape)
        return ((self.high_dimension_map(X) @ self.alpha).T @ X).T

