import numpy
import torch
import numpy as np
import PubInter

from torch import nn
from torch.utils.data import DataLoader
from torch import optim


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(     # (3, 256, 256)
            nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=2,
                stride=2,
                padding=0
            ),                          # (3, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # (3, 64, 64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 3, 2, 2, 0),   # (3, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2)             # (3, 16, 16)
        )
        self.fc1 = nn.Linear(3 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output


train_path = PubInter.get_sub_paths("D:\\train")
l = []
label_train = []
for i, p in enumerate(train_path):
    tmp = PubInter.get_mat(p)
    l.append(tmp)
    if p[9] == 'F':
        label_train.append([0.])
    else:
        label_train.append([100.])
    if i % 50 == 0:
        print("加载 train 文件:", i)

source_data = np.array(l, dtype=numpy.float32)
source_label = np.array(label_train, dtype=numpy.float32)

torch_data = GetLoader(source_data, source_label)
train_data = DataLoader(torch_data, batch_size=20, shuffle=False)


# 测试集
test_path = PubInter.get_sub_paths("D:\\test")
l = []
label_test = []
for i, p in enumerate(test_path):
    tmp = PubInter.get_mat(p)
    l.append(tmp)
    if p[8] == 'F':
        label_test.append(0)
    else:
        label_test.append(1)
    if i % 50 == 0:
        print("加载 test 文件:", i)

source_data = np.array(l, dtype=numpy.float32)
source_label = np.array(label_test, dtype=numpy.float32)

torch_data = GetLoader(source_data, source_label)
test_data = DataLoader(torch_data, batch_size=20, shuffle=False)


def accuracy(pred, labels):
    flag = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i, v in enumerate(pred):
        if v[0] >= 50 and labels[i] == 1:
            flag = flag + 1
            TP = TP + 1
        if v[0] < 50 and labels[i] == 0:
            flag = flag + 1
            TN = TN + 1
        if v[0] >= 50 and labels[i] == 0:
            FP = FP + 1
        if v[0] < 50 and labels[i] == 1:
            FN = FN + 1
    return flag, TP, TN, FP, FN


num_epochs = 32
acc_arr = []

net = CNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
for epoch in range(num_epochs):

    tmp_loss = 0
    for batch_idx, (data, target) in enumerate(train_data):
        net.train()
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tmp_loss = loss.data.numpy()

    print("epoch:", epoch, end=" ")
    print("loss:", tmp_loss, end=" ")
    net.eval()
    all_ok = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for (data_t, target_t) in test_data:
        output_t = net(data_t)
        a, b, c, d, e = accuracy(output_t, target_t)
        all_ok = all_ok + a
        TP = TP + b
        TN = TN + c
        FP = FP + d
        FN = FN + e

    acc = all_ok / (len(test_data) * 20)
    print("acc:", acc, end=" ")
    print("TP, TN, FP, FN:", TP, TN, FP, FN)
    # print("precision:", TP / (TP + FP), end=" ")
    # print("recall:", TP / (TP + FN))

    acc_arr.append(acc)

