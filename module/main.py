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

train_loader = pytorch.data_loader(train_dir)
test_loader = pytorch.data_loader(test_dir)

epochs = 2

net = pytorch.SimpleNet()

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        inputs, labels = data

        net.zero_grad()
        outs = net(inputs.view(-1, 3072))
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

'''
plt.imshow(torchvision.transforms.ToPILImage()(x))
        plt.title(labels[y])
        plt.show()
'''
