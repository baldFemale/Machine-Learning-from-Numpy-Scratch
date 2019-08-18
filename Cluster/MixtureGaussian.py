import math
import numpy as np

class MixtureGaussian:

    def __init__(self):
        self.weight = None
        self.means = None
        self.covs = None
        pass

    def distance_func(self, x1, x2):
        return np.sum([abs(x1[i]-x2[i]) for i in range(x1.shape[-1])])

    def expect(self, X, cluster_index):
        n_sample, n_feature = X.shape
        mean = self.means[cluster_index]
        cov = self.covs[cluster_index]
        ps = np.zeros(shape=(n_sample,1))
        det = np.linalg.det(cov)
        for i,sample in enumerate(X):
            coefficient = 1/(math.pow(2*math.pi,0.5*n_feature) * math.sqrt(det))
            p = coefficient * np.exp(-0.5 * (sample-mean).T @ np.linalg.inv(cov) @ (sample-mean))
            ps[i][0] = p
        return ps

    def maximum(self, X, gammas, cluster_index):
        gamma = (gammas[:,cluster_index]).reshape(-1,1)
        self.means[cluster_index] = np.sum(gamma * X, axis=0) / np.sum(gamma)
        self.covs[cluster_index] = (gamma*(X-self.means[cluster_index])).T @ \
                                   (X-self.means[cluster_index]) / np.sum(gamma)
        self.weight[cluster_index] = 1/X.shape[0] * np.sum(gamma)

    def fit(self, X, n_component, iteration):
        n_sample ,n_feature = X.shape
        self.weight = (1/n_component)*np.ones(shape=(n_component,1))
        self.means = X[[np.random.randint(0,n_sample) for i in range(n_component)]]
        self.covs = np.array([np.cov(X,rowvar=False) for i in range(n_component)])

        for epoch in range(iteration):
            gammas = np.zeros(shape=(n_sample, n_component))
            for cluster_index in range(n_component):
                gammas[:, cluster_index] = np.reshape(self.weight[cluster_index][0] * self.expect(X, cluster_index),
                                                      (n_sample,))
            temp = np.sum(gammas,axis=1).reshape(n_sample,1)
            gammas /= temp

            for cluster_index in range(n_component):
                self.maximum(X, gammas, cluster_index)

    def predict(self, X):
        n_sample = X.shape[0]
        res = np.zeros(shape=(n_sample, 1))
        for index in range(n_sample):
            sample = X[index]
            distance = float("inf")
            for c_index in range(len(self.means)):
                d = self.distance_func(sample, self.means[c_index])
                if d < distance:
                    distance = d
                    res[index] = c_index
        print(res)
        return res
