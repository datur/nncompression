from tqdm import tqdm
import unittest

from context import nncompression
from nncompression.convert import to_openvino, to_onnx
from nncompression.external.PyTorch_CIFAR10 import cifar10_models
from nncompression.utils import DEVICE
from nncompression.models.pytorch import utils as ptu
import nncompression.models.onnx.utils as onu
from nncompression.models import vgg, resnet, resnext, inception, googlenet, \
    mobilenet, shufflenet, densenet, mnasnet, alexnet

# from .context.nncompression.external.openvino.mo.utils.cli_parser import get_all_cli_parser

CIFAR_MODELS = {
    "resnet18": cifar10_models.resnet18,
    "resnet34": cifar10_models.resnet34,
    "resnet50": cifar10_models.resnet50,
    "densenet121": cifar10_models.densenet121,
    "densenet161": cifar10_models.densenet161,
    "densenet169": cifar10_models.densenet169,
    "mobilenet_v2": cifar10_models.mobilenet_v2,
    "googlenet": cifar10_models.googlenet,
    "inception_v3": cifar10_models.inception_v3,
    # "vgg_11_bn": cifar10_models.vgg11_bn,
    # "vgg_13_bn": cifar10_models.vgg13_bn,
    # "vgg_16_bn": cifar10_models.vgg16_bn,
    # "vgg_19_bn": cifar10_models.vgg19_bn
}

IMAGENET_MODELS = {
    # "resnet18": resnet.resnet18,
    # "resnet34": resnet.resnet34,
    # "resnet50": resnet.resnet50,
    # "resnet101": resnet.resnet101,
    # "resnet152": resnet.resnet152,
    # "wide_resnet50": resnet.wide_resnet50,
    # "wide_resnet101": resnet.wide_resnet101,
    # "resnext50": resnext.resnext_50, #Error tolerance failed
    # "resnext101": resnext.resnext_101,
    # "inception_v3": inception.inceptionv3, #Error Tolerance failed
    # "googlenet": googlenet.googlenet,
    "mobilenet": mobilenet.mobilenetv2,  # Error tolerance failed
    "shufflenetv2_05": shufflenet.shufflenetv2_05,
    "shufflenetv2_10": shufflenet.shufflenetv2_10,
    "densenet121": densenet.densenet121,
    "densenet161": densenet.densenet161,
    "densenet169": densenet.densenet169,
    "densenet201": densenet.densenet201,
    "mnasnet0_5": mnasnet.mnasnet0_5,
    "mnasnet1_0": mnasnet.mnasnet1_0,
    "alexnet": alexnet.alexnet,
    "vgg11": vgg.vgg11,
    "vgg11_bn": vgg.vgg11_bn,
    "vgg13": vgg.vgg13,
    "vgg13_bn": vgg.vgg13_bn,
    "vgg16": vgg.vgg16,
    "vgg16_bn": vgg.vgg16_bn,
    "vgg19": vgg.vgg19,
    "vgg19_bn": vgg.vgg19_bn
}


class TestModelConversion(unittest.TestCase):

    # def test_cifar10_to_onnx(self):
    #     test_loader = to_onnx.get_cifar_10_test_loader()
    #     dummy_data, _ = next(iter(test_loader))
    #     pbar = tqdm(CIFAR_MODELS)
    #     for m in pbar:
    #         pbar.set_description(f"{m}")
    #         net = CIFAR_MODELS[m](pretrained=True, device=DEVICE)
    #         net.eval()
    #         to_onnx.from_pytorch(net, dummy_data, f"model_conversion_tests/onnx/cifar10/{m}.onnx")

    def test_imagenet_to_onnx(self):
        val_loader = to_onnx.get_imagenet_val_loader()
        dummy_data, _ = next(iter(val_loader))
        pbar = tqdm(IMAGENET_MODELS)
        for m in pbar:
            pbar.set_description(f"{m}")

            net = IMAGENET_MODELS[m](pretrained=True)
            net.eval()
            pytorch_acc = ptu.calculate_accuracy(val_loader, net)
            print(pytorch_acc)

            onnx_model_path = f"model_conversion_tests/onnx/imagenet/{m}.onnx"
            to_onnx.from_pytorch(net, dummy_data, onnx_model_path)
            onnx_acc = onu.total_accuracy(onnx_model_path, data_loader=val_loader)

    # def test_to_openvino(self):
    #     to_openvino.from_onnx(input_model="../onnx_models/resnet50.onnx",
    #                           input_shape=[1, 3, 32, 32],
    #                           output_dir="model_conversion_tests/",
    #                           mo_path="../nncompression/external/openvino/mo.py")


if __name__ == "__main__":
    unittest.main()
