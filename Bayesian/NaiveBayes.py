import numpy as np
from collections import defaultdict

class NaiveBayes:

    def __init__(self):
        self.class_count = defaultdict(int)
        self.class_attribute_count = defaultdict(int)
        self.unique_class = None
        self.unique_attribute_class = {}
        self.n_feature = None
        self.n_sample = None
        pass

    def fit(self, X, Y):
        n_sample,n_feature = X.shape
        self.n_feature = n_feature
        self.n_sample = n_sample
        classes = np.unique(Y)
        self.unique_class = len(classes)
        for f in range(n_feature):
            self.unique_attribute_class[f] = len(np.unique(X[:,f]))
        for c in classes:
            index = np.where(Y == c)
            self.class_count[c] = len(index)
            partial_X = X[index[0]]
            for f_index in range(n_feature):
                feature_value = np.unique(partial_X[:,f_index])
                for f in feature_value:
                    self.class_attribute_count[(c,f_index,f)] = len(np.where(partial_X[:,f_index]==f))

    def cal_pro(self, sample, c):
        p = 1.0
        for i in range(self.n_feature):
            p *= 1.0*(self.class_attribute_count[(c,i,sample[i])]+1)/(self.class_count[c]+self.unique_attribute_class[i])
        p *= 1.0*(self.class_count[c]+1)/(self.n_sample+self.unique_class)
        return p

    def predict(self, X, Y):
        res = []
        for sample in X:
            res.append(np.argmax([self.cal_pro(sample,c) for c in self.class_count.keys()]))
        print(np.sum([res[i]==Y[i] for i in range(len(res))])/len(res))
        pass