import numpy as np
import struct
from LinearRegression.MultiLogitRegression import MultiLogitRegression


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


def linearregression(train_X,train_Y,test_X,test_Y):
    lcf = MultiLogitRegression(train_X, train_Y)
    lcf.fit()
    result = lcf.predict(test_X)
    print(np.sum(np.array([1 if result[i] == test_Y[i][0] else 0 for i in range(len(result))]))/len(result))


def main():
    train_X, train_Y, test_X, test_Y = load_data()
    train_X = (1/255)*(train_X.reshape((train_X.shape[0],train_X.shape[1]*train_X.shape[2])))
    train_Y = train_Y.reshape((train_Y.shape[0],1))
    test_X = (1/255)*(test_X.reshape((test_X.shape[0],test_X.shape[1]*test_X.shape[2])))
    test_Y = test_Y.reshape((test_Y.shape[0],1))
    linearregression(train_X,train_Y,test_X,test_Y)


if __name__ == '__main__':
    main()
