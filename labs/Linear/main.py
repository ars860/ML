import csv
import subprocess

import numpy as np
import numpy.linalg as lg
from numpy import ndarray


# %%

def normalize(dataset: ndarray, y: ndarray):
    y_n = np.array(y)
    [y_max, y_min] = [np.amax(y), np.amin(y)]
    for j, x in enumerate(y):
        y_n[j] = (x - y_min) / (y_max - y_min)

    dataset_n = np.array(dataset)
    [dataset_min, dataset_max] = [np.amin(dataset, axis=1), np.amax(dataset, axis=1)]
    for i, row in enumerate(dataset_n):
        for j, f in enumerate(row):
            row[j] = (f - dataset_min[i]) / (dataset_max[i] - dataset_min[i])
    return [dataset_n, y_n]


# %%

filename = 'LR/4.txt'


def readDatasetAndTest(file):
    with open(file) as f:
        m = int(next(f))
        n = int(next(f))
        d = np.empty([n, m])
        y = np.empty([n])
        for index, line in enumerate([next(f) for _ in range(n)]):
            things = line.split()
            y[index] = int(things.pop())
            d[index] = [int(x) for x in things]
        n_test = int(next(f))
        d_test = np.empty([n_test, m])
        y_test = np.empty([n_test])
        for index, line in enumerate([next(f) for _ in range(n_test)]):
            things = line.split()
            y_test[index] = int(things.pop())
            d_test[index] = [int(x) for x in things]
        return [n, m, d, y, n_test, d_test, y_test]


# [n, m, dataset, y, n_test, dataset_test, y_test] = readDatasetAndTest(filename)  # pd.read_csv(filename)

# %%

# min_max = minmax(dataset)
# [dataset, y] = normalize(dataset, y)

# %%

def findW(dataset: ndarray, y: ndarray, reg: float = 0):
    m = dataset.shape[1]
    [v, d, u] = lg.svd(dataset, full_matrices=False)
    v_tr = np.transpose(v)
    return sum([d[i] / (d[i] ** 2 + reg) * u[i].dot(v_tr[i] @ y) for i in range(m)])


# %%

def SMAPE(prediction, actual):
    return 1.0 / len(prediction) * np.sum(2 * np.abs(prediction - actual) / (np.abs(prediction) + np.abs(actual)))


# predictions_on_self = np.array([w @ x for x in dataset])
# smape_on_self = SMAPE(predictions_on_self, y)
# predictions_on_test = np.array([w @ x for x in dataset_test])
# smape_on_test = SMAPE(predictions_on_test, y_test)


# %%

def learnAngGetSMAPEOnTest(file, reg):
    [_, _, dataset, y, _, dataset_test, y_test] = readDatasetAndTest(file)
    w = findW(dataset, y, reg)
    predictions_on_self = np.array([w @ x for x in dataset])
    smape_on_self = SMAPE(predictions_on_self, y)
    predictions_on_test = np.array([w @ x for x in dataset_test])
    smape_on_test = SMAPE(predictions_on_test, y_test)

    return [smape_on_self, smape_on_test]


# %% some simply example with reg

from matplotlib import pyplot as plt

plt.title("Regulation = 0")
for i in range(7):
    file = "LR/{}.txt".format(i + 1)
    [onSelf, onTest] = learnAngGetSMAPEOnTest(file, 0)
    plt.scatter([i + 1], [onSelf], marker='o')
    plt.scatter([i + 1], [onTest], marker='^')
plt.show()

plt.title("Regulation = 1e-6")
for i in range(7):
    file = "LR/{}.txt".format(i + 1)
    [onSelf, onTest] = learnAngGetSMAPEOnTest(file, 1e-6)
    plt.scatter([i + 1], [onSelf], marker='o')
    plt.scatter([i + 1], [onTest], marker='^')
plt.show()


# %% find optimal regulation 0 .. 100

def plotOfMaxSMAPEInRange(start, end, amount, title):
    minSmape = [2.0, -1.0]
    plt.title(title)
    for reg in np.linspace(start, end, num=amount)[1:]:
        maxSmapeOnCurrent = -1

        for i in range(7):
            file = "LR/{}.txt".format(i + 1)
            [onSelf, onTest] = learnAngGetSMAPEOnTest(file, reg)
            maxSmapeOnCurrent = max(maxSmapeOnCurrent, max(onTest, onSelf))
        plt.scatter([reg], [maxSmapeOnCurrent])

        if minSmape[0] > maxSmapeOnCurrent:
            minSmape = [maxSmapeOnCurrent, reg]
    plt.show()


plotOfMaxSMAPEInRange(0, 100, 20, "Regulation to maxSmape 0 .. 100")

# %% find optimal regulation 0 .. 20

plotOfMaxSMAPEInRange(0, 20, 20, "Regulation to maxSmape 0 .. 20")

# %% find optimal regulation 10 .. 20

plotOfMaxSMAPEInRange(10, 20, 20, "Regulation to maxSmape 10 .. 20")

# %% find optimal regulation 14 .. 14.5

plotOfMaxSMAPEInRange(14, 14.5, 20, "Regulation to maxSmape 14 .. 14.5")

# %% find optimal regulation 14.24 .. 14.3

plotOfMaxSMAPEInRange(14.24, 14.3, 20, "Regulation to maxSmape 14.24 .. 14.3")

# %%

# let it be 14.285
optimalReg = 14.285

# %% test it

plt.title("Regulation = {}".format(optimalReg))
for i in range(7):
    file = "LR/{}.txt".format(i + 1)
    [onSelf, onTest] = learnAngGetSMAPEOnTest(file, optimalReg)
    plt.scatter([i + 1], [onSelf], marker='o')
    plt.scatter([i + 1], [onTest], marker='^')
plt.show()


# %% cpp stuff now

def run_cpp(file: str):
    cpp = subprocess.Popen("4_for_lab", stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    csv_answer = cpp.communicate(input=bytes(file, encoding='ascii'))[0].decode('utf-8')
    return parse_result(csv_answer)


def parse_result(csv_answer):
    as_list = list(csv.reader(csv_answer.splitlines()))
    del as_list[0]
    as_list = [(float(row[0]), float(row[1])) for row in as_list]

    self, test = [list(c) for c in zip(*as_list)]

    return [self, test]


# %% plots

def iterations2smape(file: str):
    [self_1, test_1] = run_cpp(file + "\n")

    plt.title("Iteration to SMAPE on self and test. File: {}".format(file))
    plt.scatter(range(len(self_1)), self_1, s=1, c="red", label="self")
    plt.scatter(range(len(test_1)), test_1, s=1, label="test")
    plt.legend()
    plt.show()


iterations2smape("LR/1.txt")
iterations2smape("LR/2.txt")
iterations2smape("LR/3.txt")
iterations2smape("LR/4.txt")
iterations2smape("LR/5.txt")
iterations2smape("LR/6.txt")
iterations2smape("LR/7.txt")
