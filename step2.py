import os
import numpy as np
import PubInter
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84


def get_3d(number, path):
    f = open(path)
    data = f.readlines()
    f.close()
    gps_data = []
    for line_ in data:
        line = line_[:len(line_) - 1].split(',')
        lat = float(line[0])
        lng = float(line[1])
        t = float(line[2])
        gps_data.append([lat, lng, t])

    dic = {}
    # matrix 1: geo
    mat1 = np.zeros((256, 256))
    for idx, i in enumerate(gps_data):
        x, y = PubInter.get_id(i[0], i[1])
        mat1[x][y] = 1
        h = x * 256 + y
        if dic.get(h) is None:
            dic[h] = [idx]
        else:
            dic[h].append(idx)

    # matrix 2: time difference
    mat2 = np.zeros((256, 256))
    for key, val in dic.items():
        y = key % 256
        x = int(key / 256)
        t = []
        for i in val:
            t.append(gps_data[i][2])
        mat2[x][y] = np.mean(t)

    # matrix 3: speed
    mat3 = np.zeros((256, 256))
    speed_squ = [0]
    for i in range(1, len(gps_data)):
        dis = geod.Inverse(gps_data[i][0], gps_data[i][1], gps_data[i-1][0], gps_data[i-1][1])
        v = dis['s12'] / abs(gps_data[i][2] - gps_data[i-1][2])
        speed_squ.append(v)

    for key, val in dic.items():
        y = key % 256
        x = int(key / 256)
        t = []
        for i in val:
            t.append(speed_squ[i])
        mat3[x][y] = np.median(t)

    # matrix 4 acc
    mat4 = np.zeros((256, 256))
    acc_seq = [0]
    for i in range(1, len(speed_squ)):
        acc = (speed_squ[i] - speed_squ[i-1] ) / (abs(gps_data[i][2] - gps_data[i-1][2]))
        acc_seq.append(acc)

    for key, val in dic.items():
        y = key % 256
        x = int(key / 256)
        t = []
        for i in val:
            t.append(acc_seq[i])
        mat4[x][y] = np.median(t)

    # matrix 5 direction
    mat5 = np.zeros((256, 256))
    direction = [0]
    for i in range(1, len(gps_data)):
        k = 0
        if gps_data[i][1] - gps_data[i-1][1] != 0:
            k = (gps_data[i][0] - gps_data[i-1][0]) / (gps_data[i][1] - gps_data[i-1][1])
        direction.append(k)

    for key, val in dic.items():
        y = key % 256
        x = int(key / 256)
        t = []
        for i in val:
            t.append(direction[i])
        mat5[x][y] = np.median(t)

    result = np.stack( (mat1, mat2, mat3, mat4, mat5) )
    result = np.around(result, decimals=2)
    store_path = "E:\\ADTGA_mat\\" + str(number) + ".npy"
    np.save(store_path, result)
    print("存储: ", number)


root = "E:\\ADTGA_traj"
paths = PubInter.get_sub_paths(root)

for i, p in enumerate(paths):
    get_3d(i, p)


# 根据轨迹得到相应的3d矩阵
