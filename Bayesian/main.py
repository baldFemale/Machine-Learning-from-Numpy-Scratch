from Bayesian.NaiveBayes import NaiveBayes
from sklearn import datasets
import random

def main():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    index = [random.randint(0,149) for i in range(100)]
    test_index = [i for i in range(150) if i not in index]
    clf = NaiveBayes()
    clf.fit(x[index],y[index])
    clf.predict(x[test_index],y[test_index])
    pass


if __name__ == '__main__':
    main()