# %%
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import random
from math import isqrt
from statistics import mode


def readDataset(name):
    tmp = read_csv('dataset/{}'.format(name)).values
    d = tmp[:, :-1]
    target = tmp[:, -1]
    return d, target


def compareWithTest(prediction, test_target, verbose=False):
    accuracy = 0

    for pred, actual in zip(prediction, test_target):
        if (verbose):
            print('predicted: {} actual: {}'.format(pred, actual))
        if pred == actual:
            accuracy += 1

    accuracy /= len(prediction)

    return accuracy


def tryParams(d, target, d_test, target_test, max_depth, criterion, splitter):
    test = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, splitter=splitter).fit(X=d, y=target)
    prediction = test.predict(d_test)
    return compareWithTest(prediction, target_test)


# %%

datasets_amount = 21
datasets = []
for i in range(datasets_amount):
    d, target = readDataset('{:02d}_train.csv'.format(i + 1))
    d_test, target_test = readDataset('{:02d}_test.csv'.format(i + 1))
    datasets.append([d, target, d_test, target_test])


# %%

def findBestParams(dataset_index, verbose=False):
    bestParams = (-1, -1, -1, -1)

    for max_depth in range(20):
        for criterion in ["gini", "entropy"]:
            for splitter in ["best", "random"]:
                d, target, d_test, target_test = datasets[dataset_index]
                accuracy = tryParams(d, target, d_test, target_test, max_depth + 1, criterion, splitter)
                bestParams = max(bestParams, (accuracy, max_depth + 1, criterion, splitter))
                if verbose:
                    print('maxHeight: {}, criterion: {}, splitter: {}. Accuracy: {}'
                          .format(max_depth + 1, criterion, splitter,
                                  accuracy)
                          )

    return bestParams


bestParams = []
for i in range(datasets_amount):
    bestParams.append(findBestParams(i))

# %%

highestOptimalDepth = max(enumerate(bestParams), key=lambda x: x[1][1])
lowestOptimalDepth = min(enumerate(bestParams), key=lambda x: x[1][1])
(h_i, (_, _, h_c, h_sp)) = highestOptimalDepth
(l_i, (_, _, l_c, l_sp)) = lowestOptimalDepth


def plot_depth2accuracy(dataset_index, criterion, splitter):
    x = [i + 1 for i in range(20)]
    y = []
    for max_depth in range(20):
        d, target, d_test, target_test = datasets[dataset_index]
        accuracy = tryParams(d, target, d_test, target_test, max_depth + 1, criterion, splitter)
        y.append(accuracy)

    plt.scatter(x, y)
    plt.plot(x, y)
    plt.show()


plot_depth2accuracy(h_i, h_c, h_sp)
plot_depth2accuracy(l_i, l_c, l_sp)


# %%

def growForest(dataset_index, treesAmount=1000):
    d, target, _, _ = datasets[dataset_index]

    xs, features = d.shape
    xsSqrt, featuresSqrt = isqrt(xs), isqrt(features)
    trees = []

    for i in range(treesAmount):
        columns = random.sample(range(features), featuresSqrt)
        rows = random.sample(range(xs), xsSqrt)
        d_bootstrapped, target_bootstrapped = d[:, columns][rows, :], target[rows]

        tree = DecisionTreeClassifier(max_depth=20).fit(X=d_bootstrapped, y=target_bootstrapped)
        trees.append((columns, tree))

    def classifier(d_test):
        predictions = [[] for i in range(len(d_test))]

        for columns, tree in trees:
            d_test_bootstrapped = d_test[:, columns]

            prediction = tree.predict(d_test_bootstrapped)
            for i, pred in enumerate(prediction):
                predictions[i].append(pred)

        predictions = [mode(i) for i in predictions]
        return predictions

    return classifier

# %%

# print('accuracy: {}'.format(compareWithTest(forest(d_test), target_test, verbose=True)))

forestsAccuracies = []
for i in range(datasets_amount):
    _, _, d_test, target_test = datasets[i]
    forest = growForest(i)
    forestsAccuracies.append(compareWithTest(forest(d_test), target_test))
