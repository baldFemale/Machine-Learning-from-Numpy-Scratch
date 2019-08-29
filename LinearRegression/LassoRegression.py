import numpy as np


class LassoRegression:

    def __init__(self):
        self.weight = None

    def fit(self, X, Y, lambd, iteration):
        n_sample = X.shape[0]
        ones = np.ones(shape=(n_sample,1))
        X = np.concatenate([X,ones],axis=1)
        Y = Y.reshape(-1,1)
        n_sample, n_feature = X.shape # n,m
        self.weight = np.random.normal(0.0,1.0,size=(n_feature,1))
        for epoch in range(iteration):
            for k in range(n_feature):
                temp_X = np.concatenate([X[:,:k],X[:,k+1:]],axis=1)
                if k==0:
                    temp_w = self.weight[k+1:]
                elif k==n_feature-1:
                    temp_w = self.weight[:k]
                else:
                    # print(k)
                    # print(self.weight[:k].shape)
                    # print(self.weight[k+1:].shape)
                    # print(np.concatenate([self.weight[:k],self.weight[k+1:]],axis=0))
                    temp_w = np.concatenate([self.weight[:k],self.weight[k+1:]],axis=0)
                x = X[:,k].reshape(-1,1)
                # print(Y-temp_X @ temp_w)
                px = -2*np.sum(x*(Y-temp_X @ temp_w))
                mx = 2 * np.sum(np.power(x,2))
                if px>lambd:
                    self.weight[k][0] = -(px-lambd)/mx
                elif px<lambd:
                    self.weight[k][0] = -(px+lambd)/mx
                else:
                    self.weight[k][0] = 0.0

    def predict(self, X):
        n_sample = X.shape[0]
        ones = np.ones(shape=(n_sample,1))
        X = np.concatenate([X,ones],axis=1)
        prediction = X @ self.weight
        return prediction