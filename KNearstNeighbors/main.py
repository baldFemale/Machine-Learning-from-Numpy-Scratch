import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from KNearstNeighbors.KNN import KNN
import matplotlib.pyplot as plt


def main():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    mm = MinMaxScaler()
    X = mm.fit_transform(X)
    train_index = [np.random.randint(0,50) for i in range(35)] + [np.random.randint(50,100) for i in range(35)]\
                  + [np.random.randint(100, 150) for i in range(35)]
    test_index = [i for i in range(150) if i not in train_index]
    clf = KNN()
    clf.fit(X[train_index],Y[train_index])
    figure = plt.figure()
    plt.plot([clf.predict(X[test_index], Y[test_index], k) for k in range(1,11)])
    plt.xlim(1,10)
    plt.xlabel("neighbor number")
    plt.ylabel("test accuracy")
    plt.show()


if __name__ == '__main__':
    main()