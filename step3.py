import os
import time
import PubInter
import numpy as np
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84


def get_click(t):
    h = t.tm_hour
    m = t.tm_min
    s = t.tm_sec
    return h * 3600 + m * 60 + s


def get_vec(number, path):
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

    # 1. distance
    dis_all = 0
    for i in range(1, len(gps_data)):
        d = geod.Inverse(gps_data[i][0], gps_data[i][1], gps_data[i-1][0], gps_data[i-1][1])
        dis_all = dis_all + d['s12']

    # 2. duration
    duration = abs(gps_data[len(gps_data) - 1][2] - gps_data[0][2])

    # 3. start time
    st_time = get_click( time.gmtime(gps_data[0][2]) )

    # 4. ave speed
    speed_ave = dis_all / duration

    f_ind = np.array([dis_all, duration, st_time, speed_ave])
    f_ind = np.around(f_ind, decimals=3)

    store_path = "E:\\ADTGA_ind\\" + str(number) + ".npy"
    np.save(store_path, f_ind)
    print("存储: ", number)


root = "E:\\ADTGA_traj"
paths = PubInter.get_sub_paths(root)
for i, p in enumerate(paths):
    get_vec(i, p)

