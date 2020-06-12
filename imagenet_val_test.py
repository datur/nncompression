import torch
import torchvision.transforms as transforms
import time
from nncompression.models import resnet
from nncompression.utils import DEVICE, IMAGENET_LABELS
from nncompression.models.pytorch.utils import get_imagenet_val_loader
from tqdm import tqdm


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


print_freq = 5

criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
net = resnet.resnet18(pretrained=True)
net.eval()
net.to(DEVICE)

correct = 0
total = 0

BATCH_SIZE = 1

val_loader = get_imagenet_val_loader(
    '/media/linux/imagenet', batch_size=BATCH_SIZE)

class_correct = list(0. for x in range(len(IMAGENET_LABELS)))
class_total = list(0. for x in range(len(IMAGENET_LABELS)))


print(f"Using {DEVICE.type} device")
batch_time = AverageMeter('Time', ':6.3f')
losses = AverageMeter('Loss', ':.4e')
top1 = AverageMeter('Acc@1', ':6.2f')
top5 = AverageMeter('Acc@5', ':6.2f')
progress = ProgressMeter(
    len(val_loader),
    [batch_time, losses, top1, top5],
    prefix='Test: ')


with torch.no_grad():
    end = time.time()
    for i, (images, labels) in enumerate(val_loader):

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = net(images)
        loss = criterion(outputs, labels)

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

        # for i in range(BATCH_SIZE):
        #     c = (predicted == labels).squeeze()
        #     for i in range(BATCH_SIZE):
        #         label = labels[i]
        #         class_correct[label] += c[i].item()
        #         class_total[label] += 1

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    print(top1.avg)
