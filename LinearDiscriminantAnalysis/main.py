import matplotlib.pyplot as plt
from sklearn import datasets
from LinearDiscriminantAnalysis.LDA import LDA



def main():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    lda = LDA()
    lda.fit(X,Y)
    result = lda.transform(X,n_component=2)
    figure = plt.figure()
    for i in range(len(result)):
        if Y[i] == 0:
            plt.scatter(x=result[i][0],y=result[i][1],color="red")
        if Y[i] == 1:
            plt.scatter(x=result[i][0],y=result[i][1],color="blue")
        if Y[i] == 2:
            plt.scatter(x=result[i][0],y=result[i][1],color="green")
    plt.show(figure)


if __name__ == '__main__':
    main()