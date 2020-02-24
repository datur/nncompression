"""Module containing methods to convert neural network models to onnx models.
"""
import onnx
from torch.onnx import export
from onnxruntime import InferenceSession
from ..utils import to_numpy
import numpy as np


def from_pytorch(model, dummy_data, file_name, device):
    """A function to convert a pytorch model to an onnx model. 
    Once converted perform tests to ensure model is converted correctly

    :param model: A pytorch neural network model
    :type model: torch.nn.Module
    :param dummy_data: Some data that takes the shape of the neural network inputs. 
    This doesnt have to be from training or test it can be random. ie a tensor(batch_size, input1, input2, inputn)
    :type dummy_data: tensor
    :param file_name: location for file output
    :type file_name: str
    :param device: device for pytorch compute
    :type device: torch.device
    """

    model.to(device)

    x = dummy_data.to(device)

    out = model(x)

    export(model, x, file_name, export_params=True,
           opset_version=10, do_constant_folding=True, input_names=['input'],
           output_names=['output'],
           dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}})

    onnx_model = onnx.load(file_name)
    onnx.checker.check_model(onnx_model)

    ort_session = InferenceSession(file_name)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(
        to_numpy(out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
