"""Module for converting neural network models to openvino intermediate representation
"""

import subprocess
import sys


def from_onnx(input_model: str, input_shape: list,  output_dir: str, mo_path: str):
    """
    A method for converting ONNX models into openvino IR

    :param model: [description]
    :type model: [type]

    """
    # TODO [$5e7153761b7d3d0007b9e4d9]: convert provided onnx model to openvino represnetation

    args = ["python", f"{mo_path}", "--input_model", f"{input_model}",
            "--input_shape", f"{input_shape}",
            "--output_dir", f"{output_dir}"]

    subprocess.Popen(args)
