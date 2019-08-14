import numpy as np


class Stump:

    def __init__(self):
        self.split_feature = None
        self.split_value = None
        pass

    def predict(self, X, Y):
        res = []
        for sample in X:
            if sample[self.split_feature]>=self.split_value:
                res.append(1)
            else:
                res.append(-1)
        print(np.sum([res[i]==Y[i] for i in range(len(Y))])/len(Y))
        return np.array(res).reshape(-1,1)


class Adaboost:

    def __init__(self):
        self.n_sample = None
        self.n_feature = None
        self.clfs = []
        self.weight = None
        self.clf_weight = []
        pass

    def fit(self, X, Y, classifier_number,threshold):
        self.n_sample, self.n_feature = X.shape
        self.weight = np.array([1/self.n_sample for i in range(self.n_sample)]).reshape(self.n_sample,1)
        self.clf_weight = [0 for i in range(classifier_number)]
        for i in range(classifier_number):
            clf = Stump()
            current_error = float("inf")
            for f in range(self.n_feature):
                feature = X[:,f]
                unique_value = np.unique(feature)
                for v in unique_value:
                    prediction = np.ones((self.n_sample,1))
                    prediction[feature < v] = -1
                    fake_pre = np.array([0 if prediction[i] == Y[i] else 1 for i in range(self.n_sample)]).reshape(self.n_sample,1)
                    error = np.sum(fake_pre * self.weight)
                    if error<current_error:
                        current_error = error
                        clf.split_feature = f
                        clf.split_value = v
                    if current_error<threshold:
                        break
                if current_error<threshold:
                    break
            self.clfs.append(clf)
            res = clf.predict(X, Y)
            self.clf_weight[i] = 0.5*np.log((1-error)/(error+1e-7))
            self.weight = self.weight*np.exp(Y*res.reshape(self.n_sample,1)*(-self.clf_weight[i]))
            reg = np.sum(self.weight)
            self.weight /= reg

    def predict(self, X, Y):
        ensemble_res = np.zeros((X.shape[0],1))
        for clf_index in range(len(self.clfs)):
            clf = self.clfs[clf_index]
            res = self.clf_weight[clf_index] * clf.predict(X, Y)
        ensemble_res+=res
        predict_y = np.sign(ensemble_res)
        print(np.sum([predict_y[i]==Y[i] for i in range(len(predict_y))])/len(predict_y))
