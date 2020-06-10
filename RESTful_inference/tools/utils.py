"""

"""
import torch

DEVICE = torch.device(f'c0' if torch.cuda.is_available() else 'cpu')  # nopep8 pyl disable=no-member

CIFAR10_MODELS = {}

IMAGENET_MODELS = {}


def get_top5(net, data):
    with torch.no_grad():
        data.to(DEVICE)
        out = net(data)
        return torch.topk(torch.nn.functional.softmax(out, dim=1), 5)
