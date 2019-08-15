import numpy as np
import random
import struct
from Ensemble.Adaboost import Adaboost
from Ensemble.RandomForest import RandomForest
from sklearn import datasets


def load_data():
    for file in ["train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
                 "t10k-labels.idx1-ubyte"]:
        filepath = "../dataset/"+file
        data = open(filepath,"rb").read()
        if "image" in filepath:
            fmt = ">4i"
            offset = 0
            magic_number,image_number,row,column = struct.unpack_from(fmt,data,offset)
            offset += struct.calcsize(fmt)
            fmt = ">{}B".format(row*column)
            if "train" in filepath:
                train_X = np.empty((image_number,row,column))
                for i in range(image_number):
                    train_X[i] = np.array(struct.unpack_from(fmt,data,offset)).reshape(row,column)
                    offset += struct.calcsize(fmt)
            else:
                test_X = np.empty((image_number,row,column))
                for i in range(image_number):
                    test_X[i] = np.array(struct.unpack_from(fmt,data,offset)).reshape(row,column)
                    offset += struct.calcsize(fmt)
        else:
            fmt = ">2i"
            offset = 0
            magic_number,image_number = struct.unpack_from(fmt,data,offset)
            offset += struct.calcsize(fmt)
            fmt = ">B"
            if "train" in filepath:
                train_Y = np.empty(image_number)
                for i in range(image_number):
                    train_Y[i] = struct.unpack_from(fmt,data,offset)[0]
                    offset += struct.calcsize(fmt)
            else:
                test_Y = np.empty(image_number)
                for i in range(image_number):
                    test_Y[i] = struct.unpack_from(fmt,data,offset)[0]
                    offset += struct.calcsize(fmt)
    return train_X,train_Y,test_X,test_Y

def main():
    train_X, train_Y, test_X, test_Y = load_data()
    train_X = (train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2])))
    train_Y = train_Y.reshape((train_Y.shape[0], 1))
    test_X = (test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2])))
    test_Y = test_Y.reshape((test_Y.shape[0], 1))
    index = np.where(train_Y<=1)[0]
    train_X = train_X[index]
    train_Y = train_Y[index]
    train_Y[train_Y==0] = -1
    train_index = [random.randint(0,train_X.shape[0]-1) for i in range(1000)]
    test_index = [random.randint(0,train_X.shape[0]-1) for i in range(1000)]
    # clf = Adaboost()
    # clf.fit(train_X[train_index],train_Y[train_index],10,threshold=0.1)
    # clf.predict(train_X[test_index],train_Y[test_index])

    clf = RandomForest()
    clf.fit(train_X[train_index],train_Y[train_index],100,10)
    clf.predict(train_X[test_index],train_Y[test_index])
    # iris = datasets.load_iris()
    # X = iris.data
    # Y = iris.target
    # index = np.where(Y<=1)[0]
    # X = X[index]
    # Y = Y[index]
    # Y[Y == 0] = -1
    # Y = np.expand_dims(Y,axis=1)
    # clf = Adaboost()
    # clf.fit(X,Y,10)
    # clf.predict(X,Y)


if __name__ == '__main__':
    main()