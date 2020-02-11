from network_models.utils import pytorch as ptu
from network_models import pytorch
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.functional as F
import torch.nn as nn
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

labels = ['airplane', 'automobile', 'bird',
          'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')


with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(
    f'Using {device.type} device{torch.cuda.get_device_name() if torch.cuda.is_available() else ""} on the {config["locations"]["dataset"]} dataset')


train_dir = config['locations']['train']
test_dir = config['locations']['test']


if any([config['dataset_meta']['train'][x] == None for x in config['dataset_meta']['train']]):
    print('No mean or std for training data. Calculating now...')
    config['dataset_meta']['train']['mean'], config['dataset_meta']['train']['std'] = ptu.get_normalized_dataset_vals(
        config['locations']['train'])

if any([config['dataset_meta']['test'][x] == None for x in config['dataset_meta']['test']]):
    print('No mean or std for testing data. Calculating now...')
    config['dataset_meta']['test']['mean'], config['dataset_meta']['test']['std'] = ptu.get_normalized_dataset_vals(
        config['locations']['test'])

train_loader = ptu.data_loader(
    train_dir, mean=config['dataset_meta']['train']['mean'], std=config['dataset_meta']['train']['std'])
test_loader = ptu.data_loader(
    test_dir, mean=config['dataset_meta']['test']['mean'], std=config['dataset_meta']['test']['std'])

epochs = 3

net = pytorch.DeeperCNN()

dataiter = iter(train_loader)
images, labels = dataiter.next()

net(images)

run_name = 'deeperCNN_BN'

with SummaryWriter(log_dir=f'runs/{run_name}') as w:
    w.add_graph(net, images)

net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

pytorch.train(net, train_loader, criterion, optimizer,
              epochs, device, run_name)

# net.train(train_loader=train_loader, criterion=criterion,
#           optimizer=optimizer, epochs=epochs, device=device)

# net.test()

with open('config.yaml', 'w') as f:
    yaml.dump(config, f)
