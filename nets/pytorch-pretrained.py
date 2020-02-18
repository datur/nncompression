from PyTorch_CIFAR10.cifar10_models import inception, googlenet, mobilenetv2
import torch
from network_models.utils.pytorch import data_loader
import yaml
from datasets.utils import cifar_10_labels as classes
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from torchsummary import summary

"""References

https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""


def onnx_total_accuracy(ort_session, test_loader):
    correct = 0
    total = 0
    for data in tqdm(test_loader):
        images, labels = data

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
        ort_outs = ort_session.run(None, ort_inputs)

        predicted = np.argmax(ort_outs[0], 1)
        total += labels.size(0)
        correct += (predicted == to_numpy(labels)).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def total_accuracy(test_loader, net, device):
    print('Starting test loop')
    net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def class_accuracy(test_loader, net, device):
    net = net.to(device)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'using {device}')

    print(f'Loading Dataset...')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=10,
                                              shuffle=False, num_workers=2)

    print('Creating model...')

    inception = inception.inception_v3(pretrained=True, device=device)

    print(summary(inception, (3, 32, 32)))

    inception.eval()

    # test_loader = data_loader('datasets/cifar10/test',
    #                           mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # total_accuracy(test_loader=test_loader, net=inception, device=device)

    # class_accuracy(test_loader, inception, device)

    print('start onnx conversion')
    batch_size = 10

    inception.to(device)
    images, labels = iter(test_loader).next()

    x = images.to(device)

    torch_out = inception(x)

    torch.onnx.export(inception, x, 'inceptionv3.onnx', export_params=True,
                      opset_version=10, do_constant_folding=True, input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load("inceptionv3.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("inceptionv3.onnx")

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # print(ort_outs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(
        to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    onnx_total_accuracy(ort_session, test_loader)
