import matplotlib.pyplot as plt
from sklearn import datasets
from DimensionReduce.PCA import PCA
from DimensionReduce.KPCA import KPCA


def main():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    plt.subplot(1,2,1)
    pca = PCA()
    pca.fit(X, n_component=2)
    result = pca.transform(X)
    for i in range(len(result)):
        if Y[i] == 0:
            plt.scatter(x=result[i][0],y=result[i][1],color="red")
        if Y[i] == 1:
            plt.scatter(x=result[i][0],y=result[i][1],color="blue")
        if Y[i] == 2:
            plt.scatter(x=result[i][0],y=result[i][1],color="green")
    plt.xticks([])
    plt.yticks([])
    plt.title("PCA result")
    plt.subplot(1,2,2)
    kpca = KPCA()
    kpca.fit(X, n_component=2)
    result = kpca.transform(X)
    for i in range(len(result)):
        if Y[i] == 0:
            plt.scatter(x=result[i][0],y=result[i][1],color="red")
        if Y[i] == 1:
            plt.scatter(x=result[i][0],y=result[i][1],color="blue")
        if Y[i] == 2:
            plt.scatter(x=result[i][0],y=result[i][1],color="green")
    plt.xticks([])
    plt.yticks([])
    plt.title("KPCA result")
    plt.show()


if __name__ == '__main__':
    main()

