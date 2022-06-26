import os
import datetime
from utils import get_id
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84


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


def get_sp(tr):   # tr = {[lat, lng, t], ...}

    length = len(tr)
    SP = []

    i = 0
    while i < length:

        j = i + 1
        while j < length:
            dis = geod.Inverse(tr[i][0], tr[i][1], tr[j][0], tr[j][1])
            if dis['s12'] > 200:
                delt_t = tr[j][2] - tr[i][2]
                if delt_t > 600:
                    lat = 0
                    lng = 0
                    for k in range(i, j):
                        lat = lat + tr[k][0]
                        lng = lng + tr[k][1]
                    lat = lat / (j - i)
                    lng = lng / (j - i)
                    st = tr[j][2] - tr[i][2]
                    SP.append([round(lat, 6), round(lng, 6), st])
                    i = j
                    break
            j = j + 1

        i = i + 1
    return SP


def get_click(data_):
    d = datetime.datetime.fromtimestamp(data_)
    h = d.time().hour
    m = d.time().minute
    s = d.time().second
    re = h * 3600 + m * 60 + s
    return re


def store_features(features):
    store_path = "D:\\projects\\test1\\LAB\\features256.txt"
    f = open(store_path, 'w')
    for i in features:
        f.write(str(i[0]) + "," + str(i[1]) + "," + str(i[2]) + ",")
        f.write(str(i[3]) + "," + str(i[4]) + "," + str(i[5]) + "\n")
    f.close()


def get_features(new_path):
    one_traj = get_one(new_path)

    # 1. Speed
    dis_all = 0
    for i in range(1, len(one_traj)):
        dis = geod.Inverse(one_traj[i][0], one_traj[i][1], one_traj[i - 1][0], one_traj[i - 1][1])
        dis_all = dis_all + dis['s12']

    time_all = one_traj[len(one_traj) - 1][2] - one_traj[0][2]
    ave_speed = round(dis_all / time_all, 2)
    dis_all = round(dis_all, 2)

    # print("平均速度:", ave_speed)
    # print("轨迹长度:", dis_all)
    # print("轨迹时间:", time_all)

    # 2. stay point
    SP = get_sp(one_traj)
    sp = [0, 0, 0]
    for val in SP:
        if sp[2] < val[2]:
            sp = val
    poi = 0
    if sp[0] == 0:
        poi = 0
    else:
        poi = get_id(sp[0], sp[1])
    # print("停留点:", poi)

    # 3. 起止时间

    time_st = get_click(one_traj[0][2])
    time_en = get_click(one_traj[len(one_traj) - 1][2])

    # print("开始时间: ", time_st)
    # print("结束时间: ", time_en)

    return [ave_speed, dis_all, time_all, poi, time_st, time_en]


def get_all_features():
    walk_fts = []  # 存储特征
    traj_path = "D:\\walk_path256"
    list_ = os.listdir(traj_path)
    k = 0   # 读取文件个数
    for sub_dir in list_:
        new_path = traj_path + "\\" + sub_dir
        re = get_features(new_path)
        k = k + 1
        print("读取:", k)
        walk_fts.append(re)
    print(walk_fts)

