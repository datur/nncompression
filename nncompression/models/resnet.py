"""Module containing Resnet models

"""

import torch.nn as nn


class ResNet(nn.Module):
    """Class defining the resnet neural network model


    """

    def __init__(self):
        """Initialises the model based on the input variables
        """
        super(ResNet, self).__init__()

    def forward(self, x):
        """
        Defines the forward pass of the network

        :param x: input
        :type x: tensor
        """
