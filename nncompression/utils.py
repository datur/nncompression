"""
Module containing utility functions for the Models module

"""


def to_numpy(tensor):
    """[summary]

    :param tensor: [description]
    :type tensor: [type]
    :return: [description]
    :rtype: [type]
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
