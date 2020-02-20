import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, input):
        out = self.fc1(input)
        out = F.leaky_relu(out)
        out = self.fc2(out)
        out = F.leaky_relu(out)
        out = self.fc3(out)
        out = F.leaky_relu(out)
        out = self.output(out)
        out = F.log_softmax(out, dim=1)

        return out


class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.max_pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.average_pool = nn.AvgPool2d(2, 2)

        self.fc = nn.Linear(7*7*64, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pool(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.average_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out
