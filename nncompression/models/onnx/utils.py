from tqdm import tqdm
import numpy as np
from ..utils import to_numpy


def total_accuracy(ort_session, test_loader):
    """

    :param ort_session: [description]
    :type ort_session: [type]
    :param test_loader: [description]
    :type test_loader: [type]
    """
    correct = 0
    total = 0
    for data in tqdm(test_loader):
        images, labels = data

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
        ort_outs = ort_session.run(None, ort_inputs)

        predicted = np.argmax(ort_outs[0], 1)
        total += labels.size(0)
        correct += (predicted == to_numpy(labels)).sum().item()

    return 100 * correct / total
