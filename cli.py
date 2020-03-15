"""
A Module for parsing the command line version of the program. This will define
the command line arguements for the module

"""

import click
import yaml
import torch
import torchvision
import torchvision.transforms as transforms
from nncompression.external import cifar10_models
import nncompression.models.pytorch.utils as ptu
import nncompression.models.onnx.utils as onu
from nncompression.convert import to_onnx

CIFAR_MODELS = {
    # "vgg11_bn": cifar10_models.vgg11_bn,
    # "vgg13_bn"	: cifar10_models.vgg13_bn,
    # "vgg16_bn"	: cifar10_models.vgg16_bn,
    # "vgg19_bn"	: cifar10_models.vgg19_bn,
    "resnet18"	: cifar10_models.resnet18,
    "resnet34": cifar10_models.resnet34,
    "resnet50": cifar10_models.resnet50,
    "densenet121": cifar10_models.densenet121,
    "densenet161": cifar10_models.densenet161,
    "densenet169": cifar10_models.densenet169,
    "mobilenet_v2": cifar10_models.mobilenet_v2,
    "googlenet": cifar10_models.googlenet,
    "inception_v3": cifar10_models.inception_v3
}

BATCH_SIZE = 10


def main(config, model, test_accuracy):
    """
    Main method for module

    Currently used for testing the model conversion pipeline

    :param config: [description]
    :type config: [type]
    :param model: [description]
    :type model: [type]
    """
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = config
    model_func = CIFAR_MODELS[model]

    net = model_func(pretrained=True, device=device)
    net.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    if test_accuracy:
        pyt_acc = ptu.calculate_accuracy(test_loader, net, device)
        print(f'PyTorch model accuracy: {pyt_acc}')

    dummy_data, _ = next(iter(test_loader))
    to_onnx.from_pytorch(net, dummy_data, f'{model}.onnx', device)

    if test_accuracy:
        onx_acc = onu.total_accuracy(f'{model}.onnx', test_loader)
        print(f'onnx model accuracy {onx_acc}')


def parse_yaml(cfg):
    """
    Function to load a yaml file to dictionary

    :param cfg: The config file opened
    :type cfg: file
    :return: PArsed config file
    :rtype: dict
    """
    return yaml.load(cfg, yaml.FullLoader)


@click.command()
@click.option('--model', help=f"Model choice from: {' '.join(CIFAR_MODELS.keys())}")
@click.option('--test_accuracy', help="Whether to test the mdoels accuracy with pytorch and onnx", is_flag=True)
@click.argument('config', type=click.File('r'), required=True)
def cli(model, test_accuracy, config):
    """
    Method that creates the cli for the module

    :param config: The config file containing configuration variables
    :type config: yaml file
    """

    config = parse_yaml(config.read())

    main(config, model, test_accuracy)
