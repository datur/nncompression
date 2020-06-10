"""Module containing utility functions for training and preparing data for pytorch
network model.

TODO:
Look into setting logging levels
"""

import torchvision
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ...utils import DEVICE

CIFAR10_LABELS = ['airplane', 'automobile', 'bird',
                  'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def path_data_loader(path, batch_size=10, mean=None, std=None):
    """
    A function that creates a dataloader for the root directory of a dataset split such as train, test, or validation.
    This function also applies some transforms on the data such as normalizing the data and converting the image to a tensor.

    Arguments:
        path {str} -- path to root folder of a subset of dataset ie. ../datasets/cifar10/train

    Keyword Arguments:
        batch_size {int} -- the size of each batch in the train loader (default: {10})
        mean {tuple} -- tuple containing the mean of each colour channel in the images (default: {None})
        std {tuple} -- tuple containing the standard devication of each colour channel in the dataset (default: {None})

    Returns:
        torch.utils.data.DataLoader -- an innitialised dataloader object containing len(dataset)/batch_size batches
    """
    # `torchvision.transforms.RandomPerspective(),`torchvision.transforms.RandomResizedCrop(32), torchvision.transforms.RandomRotation(10),
    transforms = [
        torchvision.transforms.ToTensor()
    ]
    if mean is not None and std is not None:
        transforms.append(torchvision.transforms.Normalize(mean, std))

    transform = torchvision.transforms.Compose(transforms)

    dataset = torchvision.datasets.ImageFolder(
        root=path, transform=transform)

    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )
    return dataset_loader


def get_normalized_dataset_vals(path):
    '''Function for computing the mean and the std of a set of data



        method created using this resource
        https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
    '''
    mean = 0.
    std = 0.
    d_loader = data_loader(path)
    for data in d_loader:
        """
        always will be data[0] as data[1] is labels
        """
        batch_samples = data[0].size(0)
        data = data[0].view(batch_samples, data[0].size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)

    mean /= len(d_loader.dataset)
    std /= len(d_loader.dataset)

    return tuple(mean.tolist()), tuple(std.tolist())


def train(net, train_loader, test_loader, criterion, optimizer, epochs, device, run_name):
    """A function for training a provided neural network.

    Arguments:
        net {nn.Module} -- Neural Network Model
        train_loader {torch.utils.data.DataLoader} -- A dataloader object for batches in training
        test_loader {torch.utils.data.DataLoader} -- A dataloader object for batches in training
        criterion {torch.nn.<lossfunction>} -- Loss Function object to facilitate training
        optimizer {torch.optim.<optimizer>} -- Optimizer object for model training
        epochs {int} -- number of epochs in training loop
        device {str} -- Specify the device the model will be trained on cpu or cuda:0
        run_name {str} -- name of the run for tensorboard tracking
    """
    print('Starting Training')
    writer = SummaryWriter(
        f'runs/{run_name}')

    running_loss = 0.0
    running_error = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            net.zero_grad()
            outs = net(inputs)
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
                    'training accuracy', running_error/500, epoch * len(train_loader) + i)

                print(
                    f'epoch: {epoch} / {epochs} | Mini batch: {i +1} / {len(train_loader)} | training_loss: {running_loss / 500} | training_error: {running_error/500}', end='\r')

                running_loss = 0.0
                running_error = 0.0
        # test loop
        writer.add_scalar('test accuracy', calculate_accuracy(
            test_loader, net, device), epoch)
    print('Finished Training')


def error_criterion(outputs, labels):
    """function for calculating the current accuracy on the network model

    Arguments:
        outputs {torch.tensor} -- a tensor containing the predicted outputs from the neural network model
        labels {torch.tensor} -- a tensor containing the ground truth labels of the predicted outputs

    Returns:
        float -- returns the calculated training error taken away from 100 to turn it to accuracy
    """
    _, max_indices = torch.max(outputs, 1)
    train_error = (max_indices != labels).float().sum()/max_indices.size()[0]
    return 100 - train_error


def calculate_accuracy(data_loader, net):
    """
    A function that calculates the accuracy of the given network model on a given testloader

    :param data_loader: iterable collection of data separated into n number of batches
    :type data_loader: torch.utils.data.DataLoader
    :param net: instance of a pytorch neural netowrk class
    :type net: torch.nn.Module
    :param device: Variable denoting the target device for model evaluation
    :type device: str
    :return: Returns the calculated accuracy metric for logging
    :rtype: float
    """
    device = DEVICE
    net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def calculate_class_accuracy(data_loader, net, device, classes) -> list:
    """
        Function for calculating the accuracy for each class in the dataset

        :param data_loader: the dataloader containing the portion of the dataset to be evaluated
        :type data_loader: torch.data
        :param net: the network to be evaluated
        :type net: torch.nn.Module
        :param device: The device for computing passes over the dataset
        :type device: torch.device
        :param classes: a list of the target classes for the network and dataset
        :type classes: list
        :return: returns a list of floats containing the accuracy of each class
        :rtype: list
        """

    net = net.to(device)
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in tqdm(data_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(classes)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    return [100 * class_correct[i] / class_total[i] for i in range(len(classes))]


def get_imagenet_val_loader(location, batch_size):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor()])

    dataset = torchvision.datasets.ImageNet(location, split="val", transform=transform)
    imgnet_val_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False)

    return imgnet_val_loader
