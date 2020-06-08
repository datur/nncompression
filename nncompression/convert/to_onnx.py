"""Module containing methods to convert neural network models to onnx models.
"""
import onnx
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import onnxruntime
import numpy as np
from ..utils import to_numpy, DEVICE, BATCH_SIZE


def from_pytorch(model, dummy_data, file_name):
    """A function to convert a pytorch model to an onnx model
    Once converted perform tests to ensure model is converted correctly

    :param model: A pytorch neural network model
    :type model: torch.nn.Module
    :param dummy_data: Some data that takes the shape of the neural network inputs. This doesnt have to be from training or test it can be random. ie a tensor(batch_size, input1, input2, inputn)
    :type dummy_data: tensor
    :param file_name: location for file output
    :type file_name: str
    :param device: device for pytorch compute
    :type device: torch.device

    """

    model.to(DEVICE)

    # print(model)

    dummy_data = dummy_data.to(DEVICE)

    out = model(dummy_data)

    torch.onnx.export(model, dummy_data, file_name, export_params=True,
                      opset_version=12, do_constant_folding=True, input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}},
                      #   operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
                      )

    onnx_model = onnx.load(file_name)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(file_name)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_data)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch resultsr
    np.testing.assert_allclose(
        to_numpy(out), ort_outs[0], rtol=1e-03, atol=2e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def get_cifar_10_test_loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)

    return test_loader


def get_imagenet_val_loader():
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    dataset = datasets.ImageNet(".", split="val", transform=transform)
    imgnet_val_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=32, shuffle=False)

    return imgnet_val_loader
