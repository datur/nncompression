"""Module for converting neural network models to openvino intermediate representation
"""

import subprocess
import sys

<<<<<<< HEAD

def from_onnx(input_model: str, input_shape: list,  output_dir: str, mo_path: str):
    """
    A method for converting ONNX models into openvino IR

    :param model: [description]
    :type model: [type]

    """
    # TODO [#16]: convert provided onnx model to openvino represnetation

    args = ["python", f"{mo_path}", "--input_model", f"{input_model}",
            "--input_shape", f"{input_shape}",
            "--output_dir", f"{output_dir}"]

    subprocess.Popen(args)
=======
def from_onnx(model):
    pass
>>>>>>> bc76a5424765457bf8784285b0362e8d48a1359a
