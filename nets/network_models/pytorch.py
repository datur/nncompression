import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def data_loader(path, batch_size=10):

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.ImageFolder(
        root=path, transform=transform)

    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )
    return dataset_loader


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

        W_out = (W_in - poolingfilter_size + 2 * poolingfilter_padding)/
                (poolingfilter_stride)                                       + 1

        so if a max pooling filter of stride 2 and width 2 and no padding the image width is halved
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 5, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.ReLU = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(in_features=256*4*4, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=10)

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.pool(out)
        out = self.ReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.pool(out)
        out = self.ReLU(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.pool(out)
        out = self.ReLU(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.tanh(out)

        x = self.fc2(out)
        return x

    def train(self, train_loader, criterion, optimizer, epochs):
        print('Starting Training')
        writer = SummaryWriter(
            f'runs/cifar10_{"simpleCNN"}_tanhlinearActivation_2fc')
        running_loss = 0.0
        for epoch in range(epochs):
            for i, data in enumerate(train_loader, 0):

                inputs, labels = data

                self.zero_grad()
                outs = self(inputs)
                loss = criterion(outs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:    # print every 100 mini-batches
                    # ...log the running loss
                    print(
                        f'epoch: {epoch} Mini batch: {i +1} / {len(train_loader)}')
                    writer.add_scalar('training loss',
                                      running_loss / 100,
                                      epoch * len(train_loader) + i)

                    writer.add_scalar('error', error_criterion(
                        outs, labels), epoch * len(train_loader) + i)
                    running_loss = 0.0

        print('Finished Training')


def error_criterion(outputs, labels):
    max_vals, max_indices = torch.max(outputs, 1)
    train_error = (max_indices != labels).float().sum()/max_indices.size()[0]
    return train_error
