import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


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

    def train(self, train_loader, criterion, optimizer, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):

                inputs, labels = data

                self.zero_grad()
                outs = self(inputs.view(-1, 3072))
                loss = criterion(outs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')


class SimpleCNN(nn.Module):
    """
    Calculating the image size after a pooling function =

        W_out = (W_in - filter_size + 2 * poolingfilter_padding)/
                (poolingfilter_stride)                                       + 1

        so if a max pooling filter of stride 2 and width 2 and no padding the image width is halved
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

        x = self.fc2_a(out)
        return x

    def train(self, train_loader, criterion, optimizer, epochs, device):
        print('Starting Training')
        writer = SummaryWriter(
            f'runs/cifar10_simpleCNN_wide_fc4')
        running_loss = 0.0
        running_error = 0.0
        for epoch in range(epochs):
            for i, data in enumerate(train_loader, 0):

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                self.zero_grad()
                outs = self(inputs)
                loss = criterion(outs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_error += error_criterion(outs, labels)

                if i % 500 == 499:    # print every 1000 mini-batches
                    # ...log the running loss

                    writer.add_scalar('training loss',
                                      running_loss / 500,
                                      epoch * len(train_loader) + i)

                    writer.add_scalar(
                        'training error', running_error/500, epoch * len(train_loader) + i)

                    print(
                        f'epoch: {epoch} / {epochs} | Mini batch: {i +1} / {len(train_loader)} | training_loss: {running_loss / 500} | training_error: {running_error/500}')

                    running_loss = 0.0
                    running_error = 0.0

        print('Finished Training')

    def test(self, test_loader):
        pass


def error_criterion(outputs, labels):
    _, max_indices = torch.max(outputs, 1)
    train_error = (max_indices != labels).float().sum()/max_indices.size()[0]
    return train_error
