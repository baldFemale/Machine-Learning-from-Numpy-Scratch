import numpy as np


class LDA:

    def __init__(self):
        self.matrix = None
        pass

    def fit(self,x,y):
        classes = np.unique(y)
        features = x.shape[1]

        # calculate within scatter
        sw = np.zeros(shape=(features,features))
        for c in classes:
            index = np.where(y == c)[0]
            cur_x = x[index]
            sw += (cur_x.shape[0]-1)*np.cov(cur_x, rowvar=0)

        # calculate between scatter
        total_mean = np.mean(x,axis=0)
        sb = np.zeros(shape=(features,features))
        for c in classes:
            index = np.where(y == c)[0]
            cur_x = x[index]
            cur_mean = np.mean(cur_x,axis=0)
            sb += cur_x.shape[0]*((cur_mean-total_mean).reshape(cur_mean.shape[0], 1) @
                                  np.transpose((cur_mean-total_mean).reshape(cur_mean.shape[0], 1)))
        self.matrix = np.linalg.inv(sw)@sb

    def transform(self,x,n_component):
        eigenvalue,eigenvector = np.linalg.eigh(self.matrix)
        index = np.argsort(eigenvalue)[::-1]
        w = eigenvector[index[:n_component]]
        return x @ w.T

