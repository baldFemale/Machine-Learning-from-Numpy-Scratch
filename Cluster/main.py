import matplotlib.pyplot as plt
from Cluster.Kmeans import Kmeans
from Cluster.MixtureGaussian import MixtureGaussian
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
    kmeans.fit(res,cluster_number=3,iteration=2000,threshold=1e-9)
    cluster_res = kmeans.predict(res,Y)
    plt.subplot(2, 1, 1)
    for i in range(len(X)):
        if cluster_res[i] == 0:
            plt.scatter(x=res[i][0],y=res[i][1],color="red")
        if cluster_res[i] == 1:
            plt.scatter(x=res[i][0],y=res[i][1],color="blue")
        if cluster_res[i] == 2:
            plt.scatter(x=res[i][0],y=res[i][1],color="green")
    plt.title("Kmeans")
    plt.xticks([])
    plt.yticks([])

    mg = MixtureGaussian()
    mg.fit(res, n_component=3, iteration=2000)
    cluster_res = mg.predict(res)
    plt.subplot(2,1,2)
    for i in range(len(X)):
        if cluster_res[i] == 0:
            plt.scatter(x=res[i][0],y=res[i][1],color="red")
        if cluster_res[i] == 1:
            plt.scatter(x=res[i][0],y=res[i][1],color="blue")
        if cluster_res[i] == 2:
            plt.scatter(x=res[i][0],y=res[i][1],color="green")
    plt.title("MixtureGaussian")
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main()
