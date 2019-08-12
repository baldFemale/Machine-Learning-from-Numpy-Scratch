import numpy as np
import cvxopt


class SupportVectorMachine:

    def __init__(self):
        self.lags = None
        self.support_vectors = None
        self.support_labels = None
        self.b = None
        pass

    def kernal_func(self, X):
        """
        :param X: shape(m,n)
        :return: shape(m,m)
        """
        return X @ X.T

    def kernal_funcII(self, x1, x2):
        # print(x1)
        # print(x1.shape)
        # print(x2.shape)
        return x1 @ x2.T

    def fit(self, X, Y, C, threshold):
        """
        optimization target: a(m,1)
        """
        n_sample, n_feature = X.shape
        P = cvxopt.matrix(Y @ Y.T * self.kernal_func(X))
        q = cvxopt.matrix(-1*np.ones(n_sample))
        G = cvxopt.matrix(np.concatenate([-np.eye(n_sample),np.eye(n_sample)],axis=0))
        h = cvxopt.matrix(np.concatenate([np.zeros(shape=(n_sample,1)),C*np.ones(shape=(n_sample,1))],axis=0))
        A = cvxopt.matrix(Y,(1,n_sample),'d')
        b = cvxopt.matrix(0.0)

        minimization = cvxopt.solvers.qp(cvxopt.matrix(P),cvxopt.matrix(q),cvxopt.matrix(G),cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))

        lags = np.array(minimization["x"])
        index = np.where(lags >= threshold)[0]
        # print(index)
        # print(X[index])
        self.lags = lags[index]
        self.support_vectors = X[index]
        self.support_labels = Y[index]
        self.b = np.sum(self.support_labels)
        # for i in range(len(self.lags)):
        #     for j in range(len(self.lags)):
        #         self.b -= self.lags[j] * self.support_labels[j] * self.kernal_funcII(self.support_vectors[j],
        #                                                                              self.support_vectors[i])
        self.b -= np.sum(self.lags*self.support_labels*self.kernal_funcII(self.support_vectors,self.support_vectors))
        self.b /= len(self.support_labels)
        # print(self.b)

    def predict(self, X, Y):
        res = []
        for sample in X:
            p = 0
            for i in range(len(self.support_vectors)):
                p += self.lags[i]*self.support_labels[i]*self.kernal_funcII(self.support_vectors[i],sample)
            p += self.b
            res.append(np.sign(p))
        print(np.sum(res[i]==Y[i] for i in range(len(res)))/len(res))

