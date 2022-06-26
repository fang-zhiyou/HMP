import os
import datetime

import numpy as np

import PubInter

from utils import get_id
from DataProcess1 import get_features
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84


def get_click(data_):
    d = datetime.datetime.fromtimestamp(data_)
    h = d.time().hour
    m = d.time().minute
    s = d.time().second
    re = h * 3600 + m * 60 + s
    return re


def get_one(path):
    f = open(path)
    all_points = f.readlines()
    f.close()
    tmp = []
    for p in all_points:
        line_ = p[:len(p) - 1].split(sep=',')
        lat = float(line_[0])
        lng = float(line_[1])
        ts = float(line_[2])
        tmp.append([lat, lng, ts])
    return tmp


def get_mat_of_two_trajectory(p1, p2, mat_p, fc_p):
    mat = []
    tr1 = get_one(p1)
    tr2 = get_one(p2)

    # 1. 获取距离矩阵 和 时间差矩阵
    mat1 = np.zeros((256, 256))
    mat3 = np.zeros((256, 256))
    for i, v1 in enumerate(tr1):
        for j, v2 in enumerate(tr2):
            dis = geod.Inverse(v1[0], v1[1], v2[0], v2[1])
            mat1[i][j] = round(dis['s12'], 2)
            mat3[i][j] = abs(get_click(v1[2]) - get_click(v2[2]))
    # print("mat1 shape:", len(mat1), len(mat1[0]))
    # print("mat3 shape:", len(mat3), len(mat3[0]))

    # 2. 获得速度差矩阵
    mat2 = np.zeros((256, 256))
    sp1 = []
    for idx in range(1, len(tr1)):
        dis = geod.Inverse(tr1[idx][0], tr1[idx][1], tr1[idx - 1][0], tr1[idx - 1][1])
        sp1.append(dis['s12'] / (tr1[idx][2] - tr1[idx - 1][2]))
    sp2 = []
    for idx in range(1, len(tr2)):
        dis = geod.Inverse(tr2[idx][0], tr2[idx][1], tr2[idx - 1][0], tr2[idx - 1][1])
        sp2.append(dis['s12'] / (tr2[idx][2] - tr2[idx - 1][2]))
    for i, v1 in enumerate(sp1):
        for j, v2 in enumerate(sp2):
            mat2[i][j] = round(abs(v1 - v2), 2)
    # print("mat2 shape:", len(mat2), len(mat2[0]))

    mat.append(mat1)
    mat.append(mat2)
    mat.append(mat3)

    # 存储结果
    f = open(mat_p, 'w')
    for idx in mat:
        for idy in idx:
            for idd, idz in enumerate(idy):
                if idd == 0:
                    f.write(str(idz))
                else:
                    f.write("," + str(idz))
            f.write('\n')
    f.close()

    f1 = get_features(p1)
    f2 = get_features(p2)
    f = open(fc_p, 'w')
    f.write(str(f1[0]) + "," + str(f1[1]) + "," + str(f1[2]) + ",")
    f.write(str(f1[3]) + "," + str(f1[4]) + "," + str(f1[5]) + "\n")
    f.write(str(f2[0]) + "," + str(f2[1]) + "," + str(f2[2]) + ",")
    f.write(str(f2[3]) + "," + str(f2[4]) + "," + str(f2[5]) + "\n")
    f.close()


if __name__ == '__main__':

    # 矩阵 [3, 256, 256]
    true_path = PubInter.get_sub_paths("D:\\walk_true")   # 100
    false_path = PubInter.get_sub_paths("D:\\taxi_false") # 400

    # # 生成训练集
    # st_mat_path = "D:\\walk_mat_train\\"
    # st_fc_path = "D:\\walk_fc_train\\"
    # for i in range(80):
    #     for j in range(i + 1, 80):
    #         sub_ = str(i) + "_to_" + str(j) + ".txt"
    #         get_mat_of_two_trajectory(true_path[i], true_path[j], st_mat_path + sub_, st_fc_path + sub_)
    #         print("训练集:", sub_)

    # 生成测试集
    st_mat_path = "D:\\taxi_mat_test\\"
    st_fc_path = "D:\\taxi_fc_test\\"
    for i in range(80):
        # for j in range(80, 100):
        #     sub_ = "T_" + str(i) + "_to_" + str(j) + ".txt"
        #     get_mat_of_two_trajectory(true_path[i], true_path[j], st_mat_path + sub_, st_fc_path + sub_)
        #     print("测试集:", sub_)

        for j in range(20):
            sub_ = "F_" + str(i) + "_to_" + str(j) + ".txt"
            get_mat_of_two_trajectory(true_path[i], false_path[j], st_mat_path + sub_, st_fc_path + sub_)
            print("测试集:", sub_)
