import torch.nn as nn
import torch.nn.functional as F


class DRE(nn.Module):
    def __init__(self, image_size):
        super(DRE, self).__init__()
        self.fc0 = nn.Linear(image_size, 500)
        self.bn0 = nn.BatchNorm1d(500)
        self.fc1 = nn.Linear(500, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500)
        self.fc5 = nn.Linear(500, 2000)
        self.bn5 = nn.BatchNorm1d(2000)
        self.fc6 = nn.Linear(2000, 500)
        self.bn6 = nn.BatchNorm1d(500)
        self.fc7 = nn.Linear(500, 100)
        self.bn7 = nn.BatchNorm1d(100)
        self.fc8 = nn.Linear(100, 2)

        self.fc0.bias.data.fill_(0)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc5.bias.data.fill_(0)
        self.fc6.bias.data.fill_(0)
        self.fc7.bias.data.fill_(0)
        self.fc8.bias.data.fill_(0)

    def forward(self, x):
        out = self.fc0(x)
        out = F.relu(self.bn0(out))
        out = self.fc1(out)
        out = F.relu(self.bn1(out))
        out = self.fc2(out)
        out = F.relu(self.bn2(out))

        out = self.fc5(out)
        dense1 = F.relu(self.bn5(out))
        out = self.fc6(dense1)
        dense2 = F.relu(self.bn6(out))
        out = self.fc7(dense2)
        dense3 = F.relu(self.bn7(out))
        out = self.fc8(dense3)

        return dense1, dense2, dense3, out


class DREFull(nn.Module):
    def __init__(self, image_size):
        super(DREFull, self).__init__()
        self.fc0 = nn.Linear(image_size, 500)
        self.bn0 = nn.BatchNorm1d(500)
        self.fc1 = nn.Linear(500, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.fc4 = nn.Linear(500, 500)
        self.bn4 = nn.BatchNorm1d(500)
        self.fc5 = nn.Linear(500, 2000)
        self.bn5 = nn.BatchNorm1d(2000)
        self.fc6 = nn.Linear(2000, 500)
        self.bn6 = nn.BatchNorm1d(500)
        self.fc7 = nn.Linear(500, 100)
        self.bn7 = nn.BatchNorm1d(100)
        self.fc8 = nn.Linear(100, 2)

        self.fc0.bias.data.fill_(0)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        self.fc4.bias.data.fill_(0)
        self.fc5.bias.data.fill_(0)
        self.fc6.bias.data.fill_(0)
        self.fc7.bias.data.fill_(0)
        self.fc8.bias.data.fill_(0)

    def forward(self, x):
        out = self.fc0(x)
        out = F.relu(self.bn0(out))
        out = self.fc1(out)
        out = F.relu(self.bn1(out))
        out = self.fc2(out)
        out = F.relu(self.bn2(out))
        out = self.fc3(out)
        out = F.relu(self.bn3(out))
        out = self.fc4(out)
        out = F.relu(self.bn4(out))

        out = self.fc5(out)
        dense1 = F.relu(self.bn5(out))
        out = self.fc6(dense1)
        dense2 = F.relu(self.bn6(out))
        out = self.fc7(dense2)
        dense3 = F.relu(self.bn7(out))
        out = self.fc8(dense3)

        return dense1, dense2, dense3, out


class DREConvLarge(nn.Module):
    def __init__(self):
        super(DREConvLarge, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.mp6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )  # 1024*3*3
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1024*3*3, 2000)
        self.bn1 = nn.BatchNorm1d(2000, affine=False)
        self.fc2 = nn.Linear(2000, 500)
        self.bn2 = nn.BatchNorm1d(500, affine=False)
        self.fc3 = nn.Linear(500, 100)
        self.bn3 = nn.BatchNorm1d(100, affine=False)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mp1(out)
        out = self.conv2(out)
        out = self.mp2(out)
        out = self.conv3(out)
        out = self.mp3(out)
        out = self.conv4(out)
        out = self.mp4(out)
        out = self.conv5(out)
        out = self.dropout(out)
        out = self.mp5(out)
        out = self.conv6(out)
        out = self.dropout(out)
        out = self.mp6(out)
        out = self.conv7(out)
        # print('out shape: ', out.shape)

        out = out.view(out.size(0), -1)
        dense1 = F.relu(self.fc1(out))

        dense2 = F.relu(self.fc2(dense1))

        dense3 = F.relu(self.fc3(dense2))

        out = self.fc4(dense3)

        return dense1, dense2, dense3, out


class DREConvSmall(nn.Module):
    def __init__(self, input_channel, input_size_y, input_size_x):
        super(DREConvSmall, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc1 = nn.Linear(64*8*8, 2000)
        self.fc1 = nn.Linear(64 * int(input_size_y / 4) * int(input_size_x / 4), 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mp1(out)
        out = self.conv2(out)
        out = self.mp2(out)
        out = self.conv3(out)
        # print('out shape: ', out.shape)

        out = out.view(out.size(0), -1)
        dense1 = F.relu(self.fc1(out))
        dense2 = F.relu(self.fc2(dense1))
        dense3 = F.relu(self.fc3(dense2))
        out = self.fc4(dense3)

        return dense1, dense2, dense3, out


class DREConvMedium(nn.Module):
    def __init__(self, input_channel, input_size_y, input_size_x):
        super(DREConvMedium, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128*int(input_size_y/8)*int(input_size_x/8), 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mp1(out)
        out = self.conv2(out)
        out = self.mp2(out)
        out = self.conv3(out)
        out = self.mp3(out)
        out = self.conv4(out)
        # print('out shape: ', out.shape)

        out = out.view(out.size(0), -1)
        dense1 = F.relu(self.fc1(out))
        dense2 = F.relu(self.fc2(dense1))
        dense3 = F.relu(self.fc3(dense2))
        out = self.fc4(dense3)

        return dense1, dense2, dense3, out


class DREConv(nn.Module):
    def __init__(self, input_channel, input_size_y, input_size_x):
        super(DREConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128*int(input_size_y/8)*int(input_size_x/8), 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mp1(out)
        out = self.conv2(out)
        out = self.mp2(out)
        out = self.conv3(out)
        out = self.mp3(out)
        out = self.conv4(out)
        # print('out shape: ', out.shape)

        out = out.view(out.size(0), -1)
        dense1 = F.relu(self.fc1(out))
        dense2 = F.relu(self.fc2(dense1))
        dense3 = F.relu(self.fc3(dense2))
        out = self.fc4(dense3)

        return dense1, dense2, dense3, out


