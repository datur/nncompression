import pickle
import numpy as np
import cv2
import os
from PIL import Image
from itertools import count
from tqdm import tqdm

CIFAR_10_LABELS = ['airplane', 'automobile', 'bird',
                   'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def cifar_10_reshape(batch_arg):
    output = np.reshape(batch_arg, (10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    return output


def cifar_to_image(file, split, out_dir):
    data = unpickle(file)
    batch_labels = data[b'labels']
    batch_images = cifar_10_reshape(data[b'data'])

    filename = ("image_%03i.jpg" % i for i in count(1))

    for image, label in zip(batch_images, batch_labels):

        path = f'{out_dir}/{split}/{CIFAR_10_LABELS[label]}/'

        if not os.path.exists(path):
            os.makedirs(path)

        img = Image.fromarray(image)
        img.save(path+next(filename))


def cifar_extract(cifar_location, out_dir):
    files = os.listdir(cifar_location)
    train_batch = [x for x in files if 'data_batch' in x]
    test_batch = [x for x in files if 'test' in x]

    for batch in tqdm(train_batch):
        batch_file = f'{file}/{batch}'
        cifar_to_image(batch_file, 'train', out_dir)

    for batch in test_batch:
        batch_file = f'{file}/{batch}'
        cifar_to_image(batch_file, 'test', out_dir)


if __name__ == "__main__":
    file = "datasets/cifar-10-batches-py/"
    out_dir = 'datasets/cifar10'
    cifar_extract(file, out_dir)
