# %%
import csv
import subprocess

import numpy as np
import pandas as pd
import numpy.linalg as lg
from numpy import ndarray
from matplotlib import pyplot as plt
from PIL import Image

filename = 'chips.csv'


def readDataset(file):
    classesNotParsed = pd.read_csv(file).values
    return [[x, y, 1 if clazz == 'P' else -1] for [x, y, clazz] in classesNotParsed]


dataset = readDataset(filename)

points = [row[:2] for row in dataset]
classes = [row[2] for row in dataset]

plt.scatter([point[0] for point in points],
            [point[1] for point in points],
            c=[('r' if clazz == -1 else 'g') for clazz in classes])
plt.show()


# %%

def prepareForCpp(dataset, C, type, args):
    argsString = type
    for arg in args:
        argsString += " {}".format(arg)

    res = "{}\n{}\n{}\n".format(len(dataset), C, argsString)
    for [x, y, clazz] in dataset:
        res += "{} {} {}\n".format(x, y, clazz)
    return res


def run_cpp(dataset, C, type, args):
    cpp = subprocess.Popen("svm_lab", stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    answer = cpp.communicate(input=bytes(prepareForCpp(dataset, C, type, args), encoding='ascii'))[0].decode('utf-8')
    return parse_result(answer)


def parse_result(answer):
    parsed = list(map(float, answer.split()))
    return [parsed[:-1], parsed[-1]]


# %%

c = 0
C = 0.5


def linear(p1, p2):
    return np.dot(p1, p2) + c


def makePolynomial(c, d, alpha):
    def polynomial(p1, p2):
        return pow(alpha * np.dot(p1, p2) + c, d)

    return polynomial


def makeExponential(beta):
    def exponential(p1, p2):
        return pow(np.e, (-beta) * np.linalg.norm(np.array(p1) - np.array(p2)))

    return exponential


def classificator(alphas, w0, kernel, x, y):
    regr = 0
    for i in range(len(points)):
        regr += alphas[i] * classes[i] * kernel(points[i], [x, y])
    regr -= w0

    return 1 if regr > 0 else -1
    # return np.sign(sum([alphas[i] * classes[i] * kernel(points[i], [x, y]) for i in range(len(dataset))]) - w0)


# %%

def drawRes(alphas, w0, imageSize, kernel, name, args):
    imagePoints = np.zeros((imageSize, imageSize, 3), 'uint8')
    linspace_x = np.linspace(-1, 1.1, imageSize) # 25
    linspace_y = np.linspace(-1, 1.1, imageSize) # 5.5

    for ix, x in enumerate(linspace_x):
        for iy, y in enumerate(linspace_y):
            prediction = classificator(alphas, w0, kernel, x, y)
            imagePoints[imageSize - 1 - iy, ix] = [0, 255, 0] if prediction == 1 else [255, 0, 0]

    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or np.math.fabs(value - array[idx - 1]) < np.math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx

    for [x, y, clazz] in dataset:
        imagePoints[imageSize - 1 - find_nearest(linspace_y, y)][find_nearest(linspace_x, x)] = \
            [255, 255, 255] if clazz == -1 else [0, 0, 0]

    im = Image.fromarray(imagePoints)
    im.save(fp="C={}_{}_{}_sz={}".format(C, name, str(args), imageSize).replace(".", "_") + ".png")
    im.show()


def plotClasses(alphas, w0, kernel):
    prediction = [classificator(alphas, w0, kernel, points[i][0], points[i][1]) for i in range(len(points))]

    plt.scatter([point[0] for point in points],
                [point[1] for point in points],
                c=['r' if p == -1 else 'g' for p in prediction])
    plt.show()


# %%

name = "linear"
c = 100 # 80 for geyser
args = [c]
C = 50 # 5 for geyser

[alphas, w0] = run_cpp(dataset, C, name, [c])
# print(alphas)
drawRes(alphas, w0, 100, linear, name, args)
plotClasses(alphas, w0, linear)

# %%

# found C = 0.5 c = 0 d = 1 [0, 1, 1]
name = "polynomial"
args = [1, 5, 0.8]
C = 0.5

[alphas, w0] = run_cpp(dataset, C, name, args)
# print(alphas)
drawRes(alphas, w0, 100, makePolynomial(*args), name, args)
plotClasses(alphas, w0, makePolynomial(*args))

# %%

name = "exponential"
args = [4.96]
C = 100

[alphas, w0] = run_cpp(dataset, C, name, args)
# print(alphas)
drawRes(alphas, w0, 100, makeExponential(*args), name, args)
plotClasses(alphas, w0, makeExponential(*args))