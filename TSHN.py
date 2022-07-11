import numpy
import torch
import random
import numpy as np
import PubInter

from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim


class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(     # (5, 256, 256)
            nn.Conv2d(
                in_channels=5,
                out_channels=5,
                kernel_size=2,
                stride=2,
                padding=0
            ),                          # (5, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # (5, 64, 64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 5, 2, 2, 0),   # (5, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2)             # (5, 16, 16)
        )
        self.fc1 = nn.Linear(5 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)

        self.fc4 = nn.Linear(4, 16)
        self.fc5 = nn.Linear(16, 4)

        self.combine = nn.Linear(132, 32)
        self.combine1 = nn.Linear(32, 1)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        x = F.relu(x)

        y = self.fc4(y)
        y = F.relu(y)
        y = self.fc5(y)
        y = F.relu(y)

        xy = torch.cat((x, y), dim=1)
        xy = self.combine(xy)
        xy = F.relu(xy)
        output = self.combine1(xy)
        return output


def load_dataset():
    paths = PubInter.get_sub_paths("E:\\np_mat")
    mat_real = []
    for p in paths:
        d = np.load(p)
        mat_real.append(d)
    mat_real = np.array(mat_real, dtype=numpy.float32)

    paths = PubInter.get_sub_paths("E:\\np_ind")
    ind_real = []
    for p in paths:
        d = np.load(p)
        ind_real.append(d)
    ind_real = np.array(ind_real, dtype=numpy.float32)

    paths = PubInter.get_sub_paths("E:\\ADTGA_mat")
    mat_false = []
    for p in paths:
        d = np.load(p)
        mat_false.append(d)
    mat_false = np.array(mat_false, dtype=numpy.float32)

    paths = PubInter.get_sub_paths("E:\\ADTGA_ind")
    ind_false = []
    for p in paths:
        d = np.load(p)
        ind_false.append(d)
    ind_false = np.array(ind_false, dtype=numpy.float32)

    # 构造 mat and ind
    mat_ind_real = []
    for i in range(100):
        mat_ind_real.append([mat_real[i], ind_real[i]])
    mat_ind_false = []
    for i in range(100):
        mat_ind_false.append([mat_false[i], ind_false[i]])

    random.shuffle(mat_ind_real)
    random.shuffle(mat_ind_false)
    tr_data = mat_ind_real[:80] + mat_ind_false[:80]
    tr_label = np.array([[100.]] * 80 + [[0.]] * 80, dtype=numpy.float32)
    t_data = mat_ind_real[80:] + mat_ind_false[80:]
    t_label = np.array([[100.]] * 20 + [[0.]] * 20, dtype=numpy.float32)

    # source_data = np.array(tr_data, dtype=numpy.float32)
    # source_label = tr_label,
    # print(source_data.shape, source_label.shape)
    torch_data = GetLoader(tr_data, tr_label)
    train_ = DataLoader(torch_data, batch_size=10, shuffle=False)

    # source_data = np.array(t_data, dtype=numpy.float32)
    # source_label = np.array(t_label, dtype=numpy.float32)
    # print(source_data.shape, source_label.shape)
    torch_data = GetLoader(t_data, t_label)
    test_ = DataLoader(torch_data, batch_size=10, shuffle=False)

    return train_, test_


train_data, test_data = load_dataset()


def accuracy(pred, label):
    a = 0
    b = 0
    c = 0
    d = 0
    for i, v in enumerate(pred):
        val = abs(v[0] - label[i][0])
        if val >= 50 and label[i][0] > 1:
            a = a + 1
        if val < 50 and label[i][0] < 1:
            b = b + 1
        if val < 50 and label[i][0] > 1:
            c = c + 1
        if val >= 50 and label[i][0] < 1:
            d = d + 1
    return a, b, c, d


num_epochs = 32

net = CNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(num_epochs):

    tmp_loss = 0
    for batch_idx, (data, target) in enumerate(train_data):
        net.train()
        output = net(data[0], data[1])
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tmp_loss = loss.data.numpy()

    print("epoch:", epoch, end=" ")
    print("loss:", tmp_loss, end=" ")
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    net.eval()
    for (data_t, target_t) in test_data:
        output_t = net(data_t[0], data_t[1])
        a, b, c, d = accuracy(output_t, target_t)
        TP = TP + a
        TN = TN + b
        FN = FN + c
        FP = FP + d
    acc = (TP + TN) / (TP + TN + FN + FP)
    print("acc:", acc, "TP:", TP, "TN:", TN, "FN:", FN, "FP:", FP)

