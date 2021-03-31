# %%
import csv
from matplotlib import pyplot as plt
import os
import subprocess

import numpy as np


def readFile(fileName):
    with open(fileName) as f:
        subject = f.readline().split()[1:]
        f.readline()
        text = f.readline().split()
        return subject, text


def readDataset(datasetFolderName):
    dirs = os.listdir(datasetFolderName)

    parts = []
    for dir in dirs:
        files = os.listdir("{}/{}".format(datasetFolderName, dir))
        part = []

        for file in files:
            subject, text = readFile("{}/{}/{}".format(datasetFolderName, dir, file))
            part.append([subject, text, 1 if "legit" in file else 0])

        parts.append(part)

    return parts


def prepareForCpp(datasetFolderName):
    parts = readDataset(datasetFolderName)

    res = "{}\n".format(len(parts))
    for part in parts:
        res += "{}\n".format(len(part))
        for [subject, text, target] in part:
            res += "{}\n{}\n{}\n{}\n{}\n".format(
                len(subject),
                ' '.join(map(str, subject)),
                len(text),
                ' '.join(map(str, text)),
                target
            )

    return res


def run_cpp_roc(file: str):
    prepared = prepareForCpp(file)

    # with open("dataset.txt", "w") as f:
    #     f.write(prepared + '/n')

    cpp = subprocess.Popen("bayes_lab_ROC.exe", stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    csv_answer = cpp.communicate(input=bytes(prepared + '\n', encoding='ascii'))[0].decode('utf-8')
    return parse_result(csv_answer)


def parse_result(csv_answer):
    as_list = list(csv.reader(csv_answer.splitlines()))
    as_list = [(float(row[0]), float(row[1])) for row in as_list]

    frp, tpr = [list(c) for c in zip(*as_list)]

    return frp, tpr

def run_cpp_lambdas(file: str):
    prepared = prepareForCpp(file)

    # with open("dataset.txt", "w") as f:
    #     f.write(prepared + '/n')

    cpp = subprocess.Popen("bayes_lab_LAMBDAS.exe", stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    csv_answer = cpp.communicate(input=bytes(prepared + '\n', encoding='ascii'))[0].decode('utf-8')
    return parse_result_lambdas(csv_answer)


def parse_result_lambdas(csv_answer):
    as_list = list(csv.reader(csv_answer.splitlines()))
    as_list = [(float(row[0]), float(row[1]), row[2] == "1") for row in as_list]

    lambdas, accs, isOk = [list(c) for c in zip(*as_list)]

    return lambdas, accs, isOk

# %%

frp, tpr = run_cpp_roc("dataset")

#%% ROC

plt.scatter(frp, tpr, s=3)
plt.plot(frp, tpr)
plt.yscale('linear')
plt.ylabel('TPR')
plt.xscale('linear')
plt.xlabel('FPR')
plt.axis([0, 1.1, 0, 1.1])
plt.show()

# %%

lambdas, accs, isOk = run_cpp_lambdas("dataset")

# %%

plt.plot(lambdas, accs)
plt.scatter(lambdas, accs, c=["r" if not i else "g" for i in isOk], s=8)
plt.ylim([0.8, 1])
plt.xscale('linear')
plt.show()