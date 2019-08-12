import numpy as np
from sklearn import datasets
from SVM.SupportVectorMachine import SupportVectorMachine
from sklearn.preprocessing import MinMaxScaler

def main():
    iris = datasets.load_iris()
    X = np.array(iris.data)
    Y = np.array(iris.target)
    index = np.where(Y<2)
    Y[Y==0] = -1
    X = X[index]
    Y = np.expand_dims(Y[index],axis=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    svm = SupportVectorMachine()
    svm.fit(X,Y,1,threshold=1e-7)
    svm.predict(X,Y)
    pass


if __name__ == '__main__':
    main()