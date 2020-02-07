from network_models.utils import pytorch as ptu
from network_models import pytorch
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.functional as F
import torch.nn as nn

labels = ['airplane', 'automobile', 'bird',
          'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_dir = 'datasets/cifar10/train'
test_dir = 'datasets/cifar10/test'

train_loader = ptu.data_loader(train_dir)
ptu.get_normalized_dataset_vals(train_loader)
test_loader = ptu.data_loader(test_dir)

# epochs = 2

# net = pytorch.SimpleCNN()
# print(net)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)

# net.train(train_loader=train_loader, criterion=criterion,
#           optimizer=optimizer, epochs=epochs)
