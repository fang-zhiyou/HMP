import os
import numpy as np


# 获得子文件
def get_sub_paths(path):
    tmp_all = []
    list_ = os.listdir(path)
    for sub_dir in list_:
        new_path = path + "\\" + sub_dir
        tmp_all.append(new_path)
    return tmp_all


def get_mat(path):
    tmp = np.loadtxt(path, delimiter=',')
    arr = tmp.reshape(3, 256, 256)
    return arr


def get_vector(path):
    tmp = np.loadtxt(path, delimiter=',')
    return tmp
