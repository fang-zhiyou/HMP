import os
import random


minLat = 39.7817
maxLat = 40.1075
minLng = 116.1715
maxLng = 116.5728
cnt = 50


def is_ok(lat, lng):
    if minLat <= lat <= maxLat and minLng <= lng <= maxLng:
        return True
    return False


def get_one_false(num):
    lat_st = random.uniform(minLat, maxLat)
    lng_st = random.uniform(minLng, maxLng)
    t_st = 0

    gps_p = [[lat_st, lng_st, t_st]]
    i = 0
    while i < cnt:
        lat_st = random.uniform(lat_st - 0.0001, lat_st + 0.0001)
        lng_st = random.uniform(lng_st - 0.0001, lng_st + 0.0001)
        t_st = t_st + random.randint(5, 10)
        if is_ok(lat_st, lng_st):
            gps_p.append([lat_st, lng_st, t_st])
            i = i + 1

    # 存储
    store_p = "E:\\ML_traj\\" + str(num) + ".txt"
    f = open(store_p, 'w')
    for gps in gps_p:
        f.write(str(gps[0]) + "," + str(gps[1]) + "," + str(gps[2]) + "\n")
    f.close()

    print("存储:", num)


for i in range(100):
    get_one_false(i)

