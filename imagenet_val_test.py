import torch

from nncompression.models import resnet
from nncompression.utils import DEVICE, IMAGENET_LABELS
from nncompression.models.pytorch.utils import get_imagenet_val_loader
from tqdm import tqdm

net = resnet.resnet18()
net.eval()
net.to(DEVICE)

correct = 0
total = 0

BATCH_SIZE = 4

val_loader = get_imagenet_val_loader('data/imagenet', batch_size=BATCH_SIZE)

class_correct = list(0. for x in range(len(IMAGENET_LABELS)))
class_total = list(0. for x in range(len(IMAGENET_LABELS)))

with torch.no_grad():
    for data in tqdm(val_loader):
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(BATCH_SIZE):
            c = (predicted == labels).squeeze()
            for i in range(BATCH_SIZE):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


print(f"Correct: {correct} \
      Total: {total}")
print(100 * correct / total)
