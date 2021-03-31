import random
from math import log, exp, copysign, isqrt

from PIL import Image
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def readDataset(file):
    classesNotParsed = pd.read_csv(file).values
    X = classesNotParsed[:, :-1]
    y = np.array(list(map(lambda c: 1 if c == 'P' else -1, classesNotParsed[:, -1:])))
    return X, y


def Q(prediction, w, X, y):
    return sum([w[i] * (1.0 if y[i] != prediction[i] else 0.0) for i in range(len(X))])


def ada(X, y, algs, T, drawer=None):
    w = np.repeat(1 / len(X), len(X))
    a = np.zeros([T])
    b = []

    precalced = np.zeros([len(algs), len(X)])

    for i, alg in enumerate(algs):
        precalced[i] = alg.predict(X)

    for t in range(T):
        Qs = np.zeros(len(algs))
        for i, alg in enumerate(algs):
            Qs[i] = Q(precalced[i], w, X, y)

        bc, bci = (Qs[0], 0)
        for i, qs in enumerate(Qs):
            bc, bci = min((bc, bci), (qs, i))

        b.append(algs[bci])
        a[t] = 0.5 * log((1 - Qs[bci]) / Qs[bci])

        for i in range(len(w)):
            w[i] = w[i] * exp(-a[t] * y[i] * b[t].predict([X[i]])[0])

        sumw = sum(w)
        for i in range(len(w)):
            w[i] = w[i] / sumw

        if drawer != None:
            def classifier(x):
                return list(map(lambda x: copysign(1, x), sum([a[i] * b[i].predict(x) for i in range(t)])))

            drawer(classifier, t)

    def classifier(x):
        return list(map(lambda x: copysign(1, x), sum([a[i] * b[i].predict(x) for i in range(T)])))

    return classifier


# %%

def plotDataset(X, y):
    plt.scatter(X[:, 0], X[:, 1],
                c=[('r' if clazz == -1 else 'g') for clazz in y])
    plt.show()


# %%

def calcAccuracy(alg, X, y):
    predictions = alg(X)
    hits = 0

    for [actual, predicted] in zip(predictions, y):
        if actual == predicted:
            hits += 1

    return hits / len(X)


def drawRes(alg, X, target, shape, imageSize, name):
    imagePoints = np.zeros((imageSize, imageSize, 3), 'uint8')
    linspace_x = np.linspace(shape[0][0], shape[0][1], imageSize)  # 25
    linspace_y = np.linspace(shape[1][0], shape[1][1], imageSize)  # 5.5

    for ix, x in enumerate(linspace_x):
        for iy, y in enumerate(linspace_y):
            prediction = alg([[x, y]])[0]
            imagePoints[imageSize - 1 - iy, ix] = [0, 255, 0] if prediction == 1 else [255, 0, 0]

    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or np.math.fabs(value - array[idx - 1]) < np.math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx

    for [[x, y], clazz] in zip(X, target):
        imagePoints[imageSize - 1 - find_nearest(linspace_y, y)][find_nearest(linspace_x, x)] = \
            [255, 255, 255] if clazz == -1 else [0, 0, 0]

    im = Image.fromarray(imagePoints)
    im.save(fp="output/{}_sz={}".format(name, imageSize).replace(".", "_") + ".png")
    im.show()


# %%

def learnAndDraw(datasetName, maxDepth, iterations, shape):
    X, y = readDataset("datasets/{}.csv".format(datasetName))

    plotDataset(X, y)

    trees = []

    xSqrt = isqrt(len(X))

    for i in range(1000):
        rows = random.sample(range(len(X)), xSqrt)
        d_bootstrapped, target_bootstrapped = X[rows, :], y[rows]

        tree = DecisionTreeClassifier(max_depth=maxDepth).fit(X=d_bootstrapped, y=target_bootstrapped)
        trees.append(tree)

    accuracies = []

    def drawer(alg, iter):
        if iter in iterations:
            accuracies.append([iter, calcAccuracy(alg, X, y)])
            drawRes(alg, X, y, shape, 100, "{}/{}_iterations_{}".format(datasetName, iter, datasetName))

    resAlg = ada(X, y, trees, 56, drawer=drawer)

    accuracies = np.array(accuracies)
    plt.scatter(accuracies[:, 0], accuracies[:, 1])
    plt.title("{}_dataset accuracies".format(datasetName))
    plt.show()

    return resAlg


# %%

iterations = [1, 2, 3, 5, 8, 13, 21, 34, 55]

# %%

geyser55 = learnAndDraw("geyser", maxDepth=3, iterations=iterations, shape=[[0, 25], [0, 5.5]])

# %%

chips55 = learnAndDraw("chips", maxDepth=3, iterations=iterations, shape=[[-1, 1.1], [-1, 1.1]])


# %%

def idealAlg(datasetName):
    X, y = readDataset("datasets/{}.csv".format(datasetName))
    tree = DecisionTreeClassifier().fit(X, y)
    return tree.predict


X_geyser, y_geyser = readDataset("datasets/{}.csv".format("geyser"))
geyserIdealAccuracy = calcAccuracy(idealAlg("geyser"), X_geyser, y_geyser)
geyserAdaAccuracy = calcAccuracy(geyser55, X_geyser, y_geyser)

X_chips, y_chips = readDataset("datasets/{}.csv".format("chips"))
chipsIdealAccuracy = calcAccuracy(idealAlg("chips"), X_chips, y_chips)
chipsAdaAccuracy = calcAccuracy(chips55, X_chips, y_chips)
