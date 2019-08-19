import numpy as np
import heapq
from collections import Counter


class kdTreeNode:

    def __init__(self, sample, target, feature_index, split_value):
        self.sample = sample
        self.target = target
        self.split_feature = feature_index
        self.split_value = split_value
        self.left = None
        self.right = None


class KNN:

    def __init__(self):
        self.n_feature = None
        self.kdTree = None
        self.heap = []
        self.K = None

    def distance_func(self, x1, x2):
        return np.sum([abs(x1[i]-x2[i]) for i in range(x1.shape[-1])])

    def construct_kdTree(self, X, Y, feature_index):
        if X is None or len(X) == 0:
            return
        if len(X) == 1:
            return kdTreeNode(X[0], Y[0], None, None)
        feature = X[:,feature_index]
        index = [(i,feature[i]) for i in range(len(X))]
        index.sort(key=lambda x: x[1])
        index.sort(key=lambda x: x[1])
        median, median_value = index[len(index)//2]
        root = kdTreeNode(X[median],Y[median], feature_index, median_value)
        root.left = self.construct_kdTree(X[:median], Y[:median], (feature_index+1) % self.n_feature)
        root.right = self.construct_kdTree(X[median+1:], Y[median+1:], (feature_index+1) % self.n_feature)
        return root

    def search_kdTree(self, sample, node):
        if not node:
            return
        if not node.left and not node.right:
            if len(self.heap)<self.K:
                heapq.heappush(self.heap,(-self.distance_func(sample,node.sample), node.target))
            else:
                d, target = heapq.heappop(self.heap)
                if -self.distance_func(sample,node.sample)>d:
                    heapq.heappush(self.heap,(-self.distance_func(sample,node.sample), node.target))
                else:
                    heapq.heappush(self.heap,(d,target))
            return
        else:
            if node.right and sample[node.split_feature]>node.split_value:
                self.search_kdTree(sample, node.right)

                if len(self.heap) < self.K:
                    heapq.heappush(self.heap, (-self.distance_func(sample, node.sample), node.target))
                else:
                    d, target = heapq.heappop(self.heap)
                    if -self.distance_func(sample, node.sample) > d:
                        heapq.heappush(self.heap, (-self.distance_func(sample, node.sample), node.target))
                    else:
                        heapq.heappush(self.heap, (d, target))

                tag = False
                if len(self.heap)<self.K:
                    tag = True
                else:
                    d, target = heapq.heappop(self.heap)
                    if node.left and -self.distance_func(sample,node.left.sample)>d:
                        tag = True
                if tag:
                    self.search_kdTree(sample, node.left)
                return
            else:
                self.search_kdTree(sample, node.left)

                if len(self.heap) < self.K:
                    heapq.heappush(self.heap, (-self.distance_func(sample, node.sample), node.target))
                else:
                    d, target = heapq.heappop(self.heap)
                    if -self.distance_func(sample, node.sample) > d:
                        heapq.heappush(self.heap, (-self.distance_func(sample, node.sample), node.target))
                    else:
                        heapq.heappush(self.heap, (d, target))

                tag = False
                if len(self.heap) < self.K:
                    tag = True
                else:
                    d, target = heapq.heappop(self.heap)
                    if node.right and -self.distance_func(sample, node.right.sample) > d:
                        tag = True
                if tag:
                    self.search_kdTree(sample, node.right)
                return

    def fit(self, X, Y):
        n_sample,n_feature = X.shape
        self.n_feature = n_feature
        self.kdTree = self.construct_kdTree(X, Y, 0)

    def predict(self, X, Y, K):
        self.K = K
        res = np.zeros(shape=Y.shape)
        for i,sample in enumerate(X):
            self.heap = []
            heapq.heapify(self.heap)
            self.search_kdTree(sample,self.kdTree)
            c = Counter(ele[1] for ele in self.heap)
            kk = None
            vv = -float("inf")
            for k,v in c.items():
                if v>vv:
                    kk = k
            res[i] = kk
        # print(res)
        # print(Y)
        return sum(res[i]==Y[i] for i in range(len(Y)))/len(Y)
