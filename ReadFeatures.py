import os
import numpy as np


def get_all_traj():
    all_traj_path_ = []
    list_ = os.listdir("D:\\walk_mat")
    for sub_dir in list_:
        new_path = "D:\\walk_mat\\" + sub_dir
        all_traj_path_.append(new_path)
    return all_traj_path_


all_path = get_all_traj()
print(all_path)

arr = np.loadtxt(all_path[0], dtype='float32', delimiter=",")
print(arr.shape)
new_arr = arr.reshape(3, 512, 512)
print(new_arr.shape)
