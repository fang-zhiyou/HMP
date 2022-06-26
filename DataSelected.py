import os
import time


minLat = 39.862142
maxLat = 40.079008 + 0.000001
minLng = 116.200058
maxLng = 116.483498 + 0.000001


user_id = "153"
label_type = 'taxi'
path_store = "D:\\taxi_false"

# 1. 得到某个用户的 label
label_data = []     # [start time, end time, label]
path_128 = "D:\\Geolife\\Data\\" + user_id + "\\labels.txt"
f = open(path_128)
data = f.readlines()
f.close()
for i in range(1, len(data)):
    line_ = data[i][0:len(data[i]) - 1].split(sep="\t")
    t1 = time.strptime(line_[0], "%Y/%m/%d %H:%M:%S")
    t2 = time.strptime(line_[1], "%Y/%m/%d %H:%M:%S")
    label_data.append([time.mktime(t1), time.mktime(t2), line_[2]])
print("label_data 长度", len(label_data), " 类型: ", label_data[0])


# 2. 得到一条轨迹
def get_traj(path):
    f = open(path)
    trs = f.readlines()[6:]
    f.close()
    traj_each = []
    for traj in trs:
        line_ = traj[:len(traj) - 1].split(sep=',')
        t = time.strptime(line_[5] + " " + line_[6], "%Y-%m-%d %H:%M:%S")
        lat = round(float(line_[0]), 6)
        lng = round(float(line_[1]), 6)
        stamp = time.mktime(t)
        traj_each.append([lat, lng, stamp])
    return traj_each


# 3. 得到所有轨迹
path_data = []  # [lat, lng, stamp, label]
traj_128 = "D:\\Geolife\\Data\\" + user_id + "\\Trajectory"
list_ = os.listdir(traj_128)
for i, sub_dir in enumerate(list_):
    new_path = traj_128 + "\\" + sub_dir
    one_traj = get_traj(new_path)
    path_data = path_data + one_traj
    if i % 50 == 0:
        print("读取:", i)


# 4. 选择 label_type
selected_label = []
for i in label_data:
    if i[2] == label_type:
        selected_label.append(i)
print("条数:", len(selected_label))

# 读取
for idx, val in enumerate(selected_label):
    st = val[0]
    en = val[1]

    # 得到路径
    walk_path = []
    for point in path_data:
        if st <= point[2] <= en:
            walk_path.append(point)

    # 抽稀 轨迹长度变成 256
    cnt = len(walk_path)
    while cnt > 256:
        tmp = []
        for i, key in enumerate(walk_path):
            if i % 2 == 0:
                tmp.append(key)
        walk_path = tmp
        cnt = len(walk_path)
    print("长度:", cnt)

    # 判断是否在北京
    flag = 0
    for i in walk_path:
        if minLat <= i[0] <= maxLat and minLng <= i[1] <= maxLng:
            flag = flag + 1
    if flag != len(walk_path):
        continue


    # 存储路径
    store_path = path_store + "\\" + str(idx) + ".txt"
    f = open(store_path, 'w')
    for i in walk_path:
        f.write(str(i[0]) + "," + str(i[1]) + "," + str(i[2]) + "\n")
    f.close()

    print("存储 walk: ", idx)


