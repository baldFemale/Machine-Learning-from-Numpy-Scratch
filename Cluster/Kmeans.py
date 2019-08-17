import numpy as np

class Kmeans:

    def __init__(self):
        self.center = None
        pass

    def distance_func(self, x1, x2):
        return np.sum([abs(x1[i]-x2[i]) for i in range(x1.shape[-1])])

    def fit(self, X, cluster_number, iteration, threshold):
        n_sample,n_feature = X.shape
        index = [np.random.randint(0,n_sample) for i in range(cluster_number)]
        self.center = X[index]

        for epoch in range(iteration):
            classes = np.zeros(shape=(n_sample,1))
            for sample_index in range(n_sample):
                sample = X[sample_index]
                distance = float("inf")
                for center_index in range(cluster_number):
                    center = self.center[center_index]
                    d = self.distance_func(sample,center)
                    if d<distance:
                        distance = d
                        classes[sample_index] = center_index
            error = 0.0
            temp_centers = np.zeros(shape=(cluster_number,n_feature))
            for c in range(cluster_number):
                temp_centers[c] = np.mean(X[np.where(classes==c)[0]],axis=0)
                error += self.distance_func(temp_centers[c],self.center[c])
            if error>threshold:
                self.center = temp_centers
            # print(self.center)

    def predict(self, X, Y):
        n_sample = X.shape[0]
        res = np.zeros(shape=(n_sample,1))
        for index in range(n_sample):
            sample = X[index]
            distance = float("inf")
            for c_index in range(len(self.center)):
                d = self.distance_func(sample,self.center[c_index])
                if d<distance:
                    distance = d
                    res[index] = c_index
        return res

