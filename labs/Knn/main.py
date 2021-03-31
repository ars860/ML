# %% helping functions

import csv
import string
import subprocess
from math import trunc

import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray


def minmax(dataset: ndarray):
    minmax = list()
    for i in range(len(dataset[0])):
        if i == len(dataset[0]) - 1:
            continue
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if i == len(row) - 1:
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset


def prepare_for_cpp_regression(dataset: ndarray, target: list, metric: string, kernel: string, window_type: string,
                               window_size: int):
    res = prepare_for_cpp_learning(dataset)

    for feature in target:
        res += str(feature) + " "
    res += " \n"

    res += metric + " \n" + kernel + " \n" + window_type + " " + str(window_size)

    return res


def run_regression(dataset: ndarray, target: list, metric: string, kernel: string, window_type: string,
                   window_size: int):
    prepared = prepare_for_cpp_regression(dataset, target, metric, kernel, window_type, window_size)
    cpp = subprocess.Popen("regression", stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(cpp.communicate(input=bytes(prepared, encoding='ascii'))[0])


def prepare_for_cpp_learning(dataset: ndarray):
    res = str(len(dataset)) + " " + str(len(dataset[0]) - 1) + " \n"
    for row in dataset:
        for i, feature in enumerate(row):
            if i == len(row) - 1:
                feature = trunc(feature)
            res += str(feature) + " "
        res += " \n"

    return res


class Result:
    metric: string
    kernel: string
    window_type: string
    window_size: float
    macroFScore: float
    microFScore: float
    isOneHot: bool

    def __str__(self) -> str:
        return self.metric + " " + self.kernel + " (" + self.window_type + ", " + str(self.window_size) + ") " + str(
            self.macroFScore) + " " + str(self.microFScore)

    def __repr__(self) -> str:
        return self.metric + " " + self.kernel + " (" + self.window_type + ", " + str(self.window_size) + ") " + str(
            self.macroFScore) + " " + str(self.microFScore)

    def __init__(self, metric: string,
                 kernel: string,
                 window_type: string,
                 window_size: float,
                 macroFScore: float,
                 microFScore: float,
                 isOneHot: bool):
        self.metric = metric
        self.kernel = kernel
        self.window_type = window_type
        self.window_size = window_size
        self.macroFScore = macroFScore
        self.microFScore = microFScore
        self.isOneHot = isOneHot


def run_learning(dataset, steps=-1):
    prepared = prepare_for_cpp_learning(dataset) + str(steps) + "\n"
    cpp = subprocess.Popen("learn", stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)

    csv_answer = cpp.communicate(input=bytes(prepared, encoding='ascii'))[0].decode('utf-8')
    return parse_result(csv_answer)


def parse_result(csv_answer):
    as_list = list(csv.reader(csv_answer.splitlines()))
    del as_list[0]
    as_list = [Result(row[0], row[1], row[2], float(row[3]), float(row[4]), float(row[5]), row[6] == 'true') for row in
               as_list]

    return as_list


# %% read dataset, normalize and pass to cpp code

filename = 'dataset.csv'

dataset = pd.read_csv(filename)

min_max = minmax(dataset.values)
dataset_normalized = normalize(dataset.values, min_max)

results = run_learning(dataset_normalized, 20)

# %% use prepared data

with open('prepared200.txt', 'r') as file:
    csv_res = file.read()

results = parse_result(csv_res)

# %% find maximums for macro and micro f scores

maximumMacroFScore = max(results, key=lambda x: x.macroFScore)
maximumMicroFScore = max(results, key=lambda x: x.microFScore)

# %% fixed windows size to macro f score plot

onlyFixedAll = list(filter(lambda res: res.window_type == "fixed", results))

fixedWindowSizeToMacroFScore = [[result.window_size, result.macroFScore] for result in onlyFixedAll]

isOneHotToColor = {
    True: "g",
    False: "r"
}

colors = [isOneHotToColor[row.isOneHot] for row in
          onlyFixedAll]

plt.scatter([x[0] for x in fixedWindowSizeToMacroFScore], [x[1] for x in fixedWindowSizeToMacroFScore],
            s=[1 for i in range(
                len(fixedWindowSizeToMacroFScore))], c=colors)

plt.show()

# %% same for micro f score

fixedWindowSizeToMicroFScore = [[result.window_size, result.microFScore] for result in onlyFixedAll]

plt.scatter([x[0] for x in fixedWindowSizeToMicroFScore], [x[1] for x in fixedWindowSizeToMicroFScore],
            s=[1 for i in range(
                len(fixedWindowSizeToMacroFScore))], c=colors)

plt.show()

# %% variable windows size to macro f score plot

onlyVariableAll = list(filter(lambda res: res.window_type == "variable", results))

variableWindowSizeToMacroFScore = [[result.window_size, result.macroFScore] for result in onlyVariableAll]

colors = [isOneHotToColor[row.isOneHot] for row in
          onlyVariableAll]

plt.scatter([x[0] for x in variableWindowSizeToMacroFScore], [x[1] for x in variableWindowSizeToMacroFScore],
            s=[1 for i in range(
                len(variableWindowSizeToMacroFScore))], c=colors)

plt.show()

# %% color for every distance taking only one hot

metricToColor = {
    "manhattan": "red",
    "euclidean": "blue",
    "chebyshev": "green"
}


def fScoreWithColors(data, colors, macro):
    if macro:
        scores = [x.macroFScore for x in data]
    else:
        scores = [x.microFScore for x in data]

    plt.scatter([x.window_size for x in data], scores,
                s=[1 for i in range(
                    len(data))], c=colors)

    plt.show()


onlyVariableOneHot = list(filter(lambda res: res.isOneHot, onlyVariableAll))

colors = [metricToColor[row.metric] for row in
          onlyVariableOneHot]

fScoreWithColors(onlyVariableOneHot, colors, True)

# %% color for every kernel taking only one hot and euclidean

onlyEuclideanOneHot = list(filter(lambda res: res.metric == "euclidean", onlyVariableOneHot))

kernelToColor = {
    "uniform": "red",
    "triangular": "blue",
    "epanechnikov": "green",
    "quartic": "black",
    "triweight": "gray",
    "tricube": "purple",
    "gaussian": "orange",
    "cosine": "aqua",
    "logistic": "pink",
    "sigmoid": "lime"
}

colors = [kernelToColor[row.kernel] for row in
          onlyEuclideanOneHot]

fScoreWithColors(onlyEuclideanOneHot, colors, True)

# %% same with micro f score

fScoreWithColors(onlyEuclideanOneHot, colors, False)

# %% only uniform now

onlyUniformOneHot = list(filter(lambda res: res.kernel == "uniform", onlyEuclideanOneHot))

plt.scatter([x.window_size for x in onlyUniformOneHot], [x.macroFScore for x in onlyUniformOneHot],
            s=[5 for i in range(
                len(onlyUniformOneHot))])
plt.plot([x.window_size for x in onlyUniformOneHot], [x.macroFScore for x in onlyUniformOneHot])

plt.show()

# %% not OneHot

notOneHottedResults = list(filter(lambda res: not res.isOneHot, results))
maximumMacroFScoreNotOneHotted = max(notOneHottedResults, key=lambda x: x.macroFScore)
maximumMicroFScoreNotOneHotted = max(notOneHottedResults, key=lambda x: x.microFScore)

# %% color for every metric

onlyVariable = list(filter(lambda res: res.window_type == "variable", notOneHottedResults))

colors = [metricToColor[row.metric] for row in
          onlyVariable]

fScoreWithColors(onlyVariable, colors, True)

# %% color for every kernel taking only eulidean metric

onlyEuclidean = list(filter(lambda res: res.metric == "euclidean", onlyVariable))

colors = [kernelToColor[row.kernel] for row in
          onlyEuclidean]

fScoreWithColors(onlyEuclidean, colors, True)

# %% same with micro

fScoreWithColors(onlyEuclidean, colors, False)

# %% optimal parameters plot

onlyTriweight = list(filter(lambda res: res.kernel == "triweight", onlyEuclidean))

plt.scatter([x.window_size for x in onlyTriweight], [x.macroFScore for x in onlyTriweight],
            s=[5 for i in range(
                len(onlyTriweight))])
plt.plot([x.window_size for x in onlyTriweight], [x.macroFScore for x in onlyTriweight])

plt.show()