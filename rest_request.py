import requests
from nncompression.models.pytorch.utils import get_imagenet_val_loader
from nncompression.utils import IMAGENET_LABELS, DEVICE
import os
import io
from io import BytesIO
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import nncompression.utils as nnutil
import base64
import json
from tqdm import tqdm
import time


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


os.environ['NO_PROXY'] = '127.0.0.1'

val_loader = get_imagenet_val_loader('data/imagenet', batch_size=1)

print(f"Using {DEVICE.type} device")
batch_time = AverageMeter('Time', ':6.3f')
top1 = AverageMeter('Acc@1', ':6.2f')
top5 = AverageMeter('Acc@5', ':6.2f')
progress = ProgressMeter(
    len(val_loader),
    [batch_time, top1, top5],
    prefix='Test: ')
print_freq = 100
class_correct = list(0. for _ in range(len(IMAGENET_LABELS)))
class_total = list(0. for _ in range(len(IMAGENET_LABELS)))

with torch.no_grad():
    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        
        '''
        TODO: Experiment with json to binary to deliver the tensor as a list
        TODO: Record the results somewhere maybe in a json file or a database or tensorboard
        TODO: Average meter for latency inbound and outbound - this should be the time arrived on server minus time sent. and same for response.
        TODO: work out accuracy per class
        '''

        # load image into pil format
        im = transforms.ToPILImage()(images[0])

        # convert pil image to bytes
        with BytesIO() as output:
            im.save(output, 'JPEG')
            data = output.getvalue()

        url = "http://ml-inference.northeurope.cloudapp.azure.com:5000/inference"

        payload = {
            'files': (
                '1.jpeg',
                data,
                'image/jpeg'
            )
        }

        r = requests.post(url, files=payload)
        r_json = r.json()

        outputs = torch.tensor(
            r_json['result']['prediction_raw']).unsqueeze(0)

        outputs, labels = outputs.to(DEVICE), labels.to(DEVICE)

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    print(batch_time.avg)
    print(top1.avg)
