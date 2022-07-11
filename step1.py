import os
import PubInter

# 116.171428, 39.781765
# 116.572719, 40.107469
minLat = 39.7817
maxLat = 40.1075
minLng = 116.1715
maxLng = 116.5728

folder = "E:\\128\\Trajectory"
label_path = "E:\\128\\labels.txt"

paths = PubInter.get_sub_paths(folder)
gps_points = []
for i, p in enumerate(paths):
    tmp = PubInter.get_traj(p)
    gps_points = gps_points + tmp
    print("读取: ", i)

label_data = PubInter.get_labels(label_path)
# walk: 174     car:452     bike: 60    train: 38


for i, item in enumerate(label_data):
    if item[2] != 'car':
        continue

    # 获取一条轨迹的所有点
    one_path_points = []
    for point in gps_points:
        if item[0] <= point[2] <= item[1]:
            one_path_points.append(point)

    # 判断在不在区间内
    flag = 0
    for point in one_path_points:
        if minLat <= point[0] <= maxLat and minLng <= point[1] <= maxLng:
            flag = flag + 1
    if flag != len(one_path_points):
        continue

    # 存储点
    walk_path_store = "E:\\car_traj\\" + str(i) + ".txt"
    f = open(walk_path_store, 'w')
    for point in one_path_points:
        f.write(str(point[0]) + "," + str(point[1]) + "," + str(point[2]) + "\n")
    f.close()

    print("存储 car: ", i)
