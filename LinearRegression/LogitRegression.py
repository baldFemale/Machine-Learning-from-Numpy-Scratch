import numpy as np


class LogitRegression:

    def __init__(self, x, y, learning_rate, epoch):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.epoch = epoch
        temp = np.ones((self.x.shape[0], 1))
        self.x = np.concatenate((self.x, temp),axis=1)
        self.theta = np.random.normal(loc=0.0, scale=1.0, size=(self.x.shape[1], 1))
        self.x_transpose = np.transpose(self.x)

    def cal_gradient(self):
        temp = (1/(1+np.exp(-np.matmul(self.x, self.theta))))-self.y
        gradients = np.matmul(self.x_transpose, temp)
        return gradients

    def fit(self):
        for step in range(self.epoch):
            gradients = (1/self.x.shape[0])*self.cal_gradient()
            self.theta -= gradients*self.learning_rate

    def predict(self,test_x):
        test_x = np.concatenate((test_x,np.ones((test_x.shape[0],1))),axis=1)
        return 1/(1+np.exp(-(test_x @ self.theta)))

