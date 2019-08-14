import numpy as np


class DecisionTreeNode:

    def __init__(self, feature_index=None, split_value=None, left_tree=None, right_tree=None, value=None):
        self.feature_index = feature_index
        self.split_value = split_value
        self.left_tree = left_tree
        self.right_tree = right_tree
        self.value = value


class DecisionTree:

    def __init__(self, max_depth, min_sample, min_gain):
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.min_gain = min_gain
        self.root = None

    def cal_entropy(self, y):
        entropy = 0
        unique_label = np.unique(y)
        for label in unique_label:
            p = len(y[y == label])/len(y)
            entropy += -p*np.log2(p)
        return entropy

    def cal_gain(self, y, y1, y2):
        p = len(y1)/len(y)
        gain = self.cal_entropy(y)-p*self.cal_entropy(y1)-(1-p)*self.cal_entropy(y2)
        return gain

    def split(self, XY, feature_index, value):
        split_func = lambda x: x[feature_index] >= value
        left = np.array([sample for sample in XY if split_func(sample)])
        right = np.array([sample for sample in XY if not split_func(sample)])
        return left, right

    def vote4most(self,y):
        unique_label = np.unique(y)
        most_common_label = None
        most_common_count = 0
        for label in unique_label:
            l = len(y[y == label])
            if l>most_common_count:
                most_common_count = l
                most_common_label = label
        return most_common_label

    def build_tree(self, X, Y, current_depth):
        largest_gain = 0
        XY = np.concatenate([X,Y],axis=1)
        n_sample,n_feature = X.shape
        if n_sample>=self.min_sample and current_depth<=self.max_depth:
            for feature_index in range(n_feature):
                feature_value = X[:,feature_index]
                unique_values = np.unique(feature_value)
                for unique_value in unique_values:
                    xy1, xy2 = self.split(XY,feature_index,unique_value)
                    if len(xy1)>=1 and len(xy2)>=1:
                        y1 = xy1[:,-1]
                        y2 = xy2[:,-1]
                        info_gain = self.cal_gain(Y, y1, y2)
                        if info_gain>=largest_gain:
                            largest_gain = info_gain
                            best_split = {"feature_index":feature_index,"split_value":unique_value}
                            best_sets = {
                                "left_X": xy1[:,:n_feature],
                                "right_X": xy2[:,:n_feature],
                                "left_Y": xy1[:,-1],
                                "right_Y": xy2[:,-1]
                            }
        if largest_gain>self.min_gain:
            left_tree = self.build_tree(best_sets["left_X"],np.expand_dims(best_sets["left_Y"],axis=1),current_depth+1)
            right_tree = self.build_tree(best_sets["right_X"],np.expand_dims(best_sets["right_Y"],axis=1),current_depth+1)
            return DecisionTreeNode(feature_index=best_split["feature_index"],split_value=best_split["split_value"],
                                    left_tree=left_tree,right_tree=right_tree)
        return DecisionTreeNode(value=self.vote4most(Y))

    def fit(self, X, Y):
        print("start fitting")
        self.root = self.build_tree(X, Y, 0)

    def predict(self, x, root=None):
        if not root:
            root = self.root

        if root.value is not None:
            return root.value

        feature_index = root.feature_index
        x_value = x[feature_index]
        if x_value >= root.split_value:
            node = root.left_tree
        else:
            node = root.right_tree
        return self.predict(x, node)

    def predict_all(self, X, Y):
        temp = np.array([self.predict(sample) for sample in X])
        print(np.sum([temp[i]==Y[i] for i in range(len(temp))])/len(temp))
        return temp






