# Masters Thesis

## Neural Network Compression

Current State of Neural Network Compression and Deployment

![](diagram/diagram.gv.png)

## Tools:

### Model Description Libraries:

| Library                                                     | Maintainer                                    | Desc                                                                                                          | Status             |
| ----------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------ |
| [PyTorch](https://pytorch.org/)                             | Facebook                                      | Facebook backed and Open Source. Dynamic Computational graph                                                  | Actively Developed |
| [Tensorflow](https://www.tensorflow.org/)                   | Google                                        | Google Backed. Tensorflow Lite for Mobile Deployment. Python and C++. Static graph                            | Actively Developed |
| [Keras](https://keras.io/)                                  |                                               | Google Employee François Cholle developed. Great for quick prototyping. Wrapper for Tensorflow, Theano, CNTK. | Active             |
| [CNTK](https://docs.microsoft.com/en-gb/cognitive-toolkit/) | Microsoft                                     | Microsoft backed.                                                                                             | Depreciated        |
| [ONNX](https://onnx.ai/)                                    | Joint Venture between Facebook and Microsoft. | Wide spread enterprise use. Multiple libraries convert to universal Model. Interchangable AI Models           | Actively Developed |
| [MxNet](https://mxnet.apache.org/)                          | Apache                                        |                                                                                                               | Actively Developed |
| [Chainer](https://chainer.org/)                             |                                               |                                                                                                               | Actively Developed |
| [Caffe](https://caffe.berkeleyvision.org/)                  | UC Berkley                                    |                                                                                                               | Depreciated        |
| [Caffe2](https://caffe2.ai/)                                | Merged with PyTorch as of 2018                |                                                                                                               | Depreciated        |
| [Theano](http://deeplearning.net/software/theano/)          |                                               |                                                                                                               | Depreciated        |

### Compression Libraries and extensions:

| Tool                                                                                                                                                    | Desc                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [FINN](https://xilinx.github.io/finn/)                                                                                                                  |
| [Vitis](https://www.xilinx.com/products/design-tools/vitis/vitis-platform.html)                                                                         |
| [Intel Distiller](https://github.com/NervanaSystems/distiller)                                                                                          | Compression Library ontop of PyTorch. State of the art algorithms. Includes Pruning, Quantisation, Regularization, Knowledge Distilation, Conditional Computation |
| [Keras-Surgeon](https://github.com/BenWhetton/keras-surgeon)                                                                                            |
| [QNNPACK](https://github.com/pytorch/QNNPACK)                                                                                                           |
| [MXNet Contrib Quantization](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Graph+Optimization+and+Quantization+based+on+subgraph+and+MKL-DNN) |
| [tensorflow_model_optimization](https://www.tensorflow.org/model_optimization)                                                                          |
| [Chainer Pruner](https://github.com/DeNA/ChainerPruner)                                                                                                 |

### Deployment Tools:

| Tool                                                          | Maintainer | Desc                                       |
| ------------------------------------------------------------- | ---------- | ------------------------------------------ |
| [TVM Neural Network Compiler Stack](https://tvm.ai/)          | Apache     | Reay. VTA                                  |
| [Glow](https://github.com/pytorch/glow)                       | Facebook   | Pytorch Compiler                           |
| [OpenVino](https://software.intel.com/en-us/openvino-toolkit) | Intel      | Deployment and model optimization for FPGA |

## References
