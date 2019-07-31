import numpy as np
from LinearRegression.LogitRegression import LogitRegression


class MultiLogitRegression:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.classes = np.unique(self.y)
        self.index = {}
        for i in self.classes:
            self.index[i] = np.where(y == i)[0]
        self.classifier = {}

    def fit(self):
        for i in range(len(self.classes)):
            for j in range(i+1,len(self.classes)):
                print("modeling {}th and {}th class".format(i,j))
                x = np.concatenate((self.x[self.index[self.classes[i]]],self.x[self.index[self.classes[j]]]),axis=0)
                y = np.concatenate((np.ones(shape=self.index[self.classes[i]].shape),
                                    np.zeros(shape=self.index[self.classes[j]].shape)), axis=0)
                y = y.reshape((y.shape[0],1))
                lcf = LogitRegression(x, y, 0.001, 5000)
                lcf.fit()
                self.classifier[(i,j)] = lcf

    def predict(self, x):
        temp = np.zeros(shape=(len(self.classifier),x.shape[0]))
        for k,lcf in self.classifier.items():
            i,j = k
            result = lcf.predict(x)
            result = result.reshape((result.shape[0],))
            temp[i] += result
            temp[j] += (1-result)
        class_index = np.argmax(temp,axis=0)
        return np.array([self.classes[i] for i in class_index])
