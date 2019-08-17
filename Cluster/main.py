import matplotlib.pyplot as plt
from Cluster.Kmeans import Kmeans
from LinearDiscriminantAnalysis.LDA import LDA
from sklearn import datasets
from sklearn import preprocessing


def main():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    mm = preprocessing.MinMaxScaler()
    mm.fit_transform(X)
    lda = LDA()
    lda.fit(X,Y)
    res = lda.transform(X,2)
    kmeans = Kmeans()
    kmeans.fit(res,cluster_number=3,iteration=1000,threshold=1e-9)
    cluster_res = kmeans.predict(res,Y)
    plt.figure()
    for i in range(len(X)):
        if cluster_res[i] == 0:
            plt.scatter(x=res[i][0],y=res[i][1],color="red")
        if cluster_res[i] == 1:
            plt.scatter(x=res[i][0],y=res[i][1],color="blue")
        if cluster_res[i] == 2:
            plt.scatter(x=res[i][0],y=res[i][1],color="green")
    plt.show()


if __name__ == '__main__':
    main()
