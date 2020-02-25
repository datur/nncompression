"""
Module containing utility functions relating to onnx models

"""
from tqdm import tqdm
import numpy as np
import onnxruntime
from ...utils import to_numpy


def total_accuracy(onnx_model, data_loader):
    """
    Function for calculating the accuracy of an onnx model

    :param ort_session: onnxruntime session object preinitialised to the specified model
    :type ort_session: onnxruntime.InferenceSession
    :param data_loader: a dataloader object for a dataset split for evaluation
    :type data_loader: torch.data.DataLoader
    :return: The calculated accuracy of the model on the provided dataloader
    :rtype: float
    """
    ort_session = onnxruntime.InferenceSession(onnx_model)
    correct = 0
    total = 0
    for data in tqdm(data_loader):
        images, labels = data

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
        ort_outs = ort_session.run(None, ort_inputs)

        predicted = np.argmax(ort_outs[0], 1)
        total += labels.size(0)
        correct += (predicted == to_numpy(labels)).sum().item()

    return 100 * correct / total
