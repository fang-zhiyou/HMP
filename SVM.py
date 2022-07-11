import random

import numpy as np
import PubInter
import joblib
from sklearn import svm
import matplotlib.pyplot as plt


def accuracy(pred, label):
    a = 0
    b = 0
    c = 0
    d = 0
    for i, v in enumerate(pred):
        if v == label[i] and v == 1:
            a = a + 1
        if v == label[i] and v == 0:
            b = b + 1
        if v != label[i] and v == 0:
            c = c + 1
        if v != label[i] and v == 1:
            d = d + 1
    return a, b, c, d


def load_data():
    paths = PubInter.get_sub_paths("E:\\np_ind")
    ind_real = []
    for p in paths:
        d = np.load(p)
        ind_real.append(d)

    paths = PubInter.get_sub_paths("E:\\car_ind")
    ind_false = []
    for p in paths:
        d = np.load(p)
        ind_false.append(d)

    random.shuffle(ind_real)
    random.shuffle(ind_false)

    x_tr = ind_real[:80] + ind_false[:80]
    x_tr = np.array(x_tr)
    y_tr = [1] * 80 + [0] * 80
    y_tr = np.array(y_tr)

    x_t = ind_real[80:] + ind_false[80:]
    x_t = np.array(x_t)
    y_t = [1] * 20 + [0] * 20
    y_t = np.array(y_t)

    return x_tr, y_tr, x_t, y_t


num_epochs = 32
for epoh in range(num_epochs):
    x_train, y_train, x_test, y_test = load_data()

    # train
    model = svm.SVC(C=1, kernel='linear')
    model.fit(x_train, y_train)

    # predict
    output = model.predict(x_test)

    TP, TN, FN, FP = accuracy(output, y_test)
    acc = (TP + TN) / (TP + TN + FN + FP)
    print("epoch:", epoh, end=" ")
    print("acc:", acc, "TP:", TP, "TN:", TN, "FN:", FN, "FP:", FP)

