"""Simple CNN

This module contains a class defining a simple CNN in pytorch


"""
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    This class defines a simple CNN with 3 Convolutional layers and one hidden
    linear layer before output layer of 10
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.layer_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU()
        )

        self.layer_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU()
        )

        self.layer_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU()
        )

        self.fc1_a = nn.Sequential(
            nn.Linear(128*4*4, 100),
            nn.LeakyReLU()
        )
        self.fc2_a = nn.Sequential(
            nn.Linear(100, 10)
        )

    def forward(self, input):

        out = self.layer_conv1(input)

        out = self.layer_conv2(out)

        out = self.layer_conv3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1_a(out)

        out = self.fc2_a(out)
        return out
