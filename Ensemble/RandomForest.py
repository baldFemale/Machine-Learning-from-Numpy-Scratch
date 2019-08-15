import numpy as np
import random
from Ensemble.Adaboost import Stump
import matplotlib.pyplot as plt


class RandomForest:

    def __init__(self):
        self.clfs = []
        pass

    def bootstrap(self, X, m):
        n_sample = X.shape[0]
        index = [random.randint(0,n_sample-1) for i in range(m)]
        return index

    def fit(self, X, Y, n_models,k_feature):
        n_sample, n_feature = X.shape
        for i in range(n_models):
            clf = Stump()
            bootstrap_index = self.bootstrap(X,int(0.1*n_sample))
            input_X, input_Y = X[bootstrap_index], Y[bootstrap_index]
            feature_index = random.sample(range(0,n_feature-1),k_feature)
            min_error = float("inf")
            for f in feature_index:
                unique_values = np.unique(input_X[:,f])
                for v in unique_values:
                    prediction = np.ones(input_Y.shape)
                    prediction[input_X[:, f] <= v] = -1
                    error = np.sum([prediction[i]!=input_Y[i] for i in range(len(input_Y))])/len(input_Y)
                    if error < min_error:
                        min_error = error
                        clf.split_feature = f
                        clf.split_value = v
            self.clfs.append(clf)

    def predict(self, X, Y):
        prediction = np.zeros(Y.shape)
        basic_res = []
        for clf in self.clfs:
            res = clf.predict(X, Y)
            prediction+=res
            basic_res.append(np.sum([res[i]==Y[i] for i in range(len(Y))])/len(Y))
        prediction = np.sign(prediction)
        # print(np.sum([prediction[i]==Y[i] for i in range(len(Y))])/len(Y))
        figure = plt.figure()
        plt.plot(basic_res,color="red",label="basic_classifier")
        plt.plot([np.sum([prediction[i]==Y[i] for i in range(len(Y))])/len(Y)]*(len(basic_res)),color="blue",
                 label="random_forest")
        plt.legend()
        plt.show()
