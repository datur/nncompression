# Masters Thesis

## Neural Network Compression

Current State of Neural Network Compression and Deployment

![](diagram/diagram.gv.png)

### Tools:

Model Description Libraries:

- PyTorch
  - Facebook backed and Open Source.
  - Dynamic Computational graph
- Tensorflow
  - Google Backed. Tensorflow Lite for Mobile Deployment. Python and C++.
  - Static graph
- Keras
  - Google Employee François Cholle developed. Great for quick prototyping. Wrapper for Tensorflow, Theano, CNTK.
- CNTK
  - Microsoft backed. Depreciated.
- ONNX
  - Joint Venture between Facebook and Microsoft. Wide spread enterprise use.
  - Multiple libraries convert to universal Model
  - Interchangable AI Models
- MxNet
  - Apache.
- Chainer
- Caffe
  - UC Berkley Project. Depreciated
- Caffe2
  - Merged with PyTorch as of 2017.
- Theano
  - Depreciated

Compression Libraries and extensions:

- FINN [Link](https://xilinx.github.io/finn/)
- Vitis [Link](https://www.xilinx.com/products/design-tools/vitis/vitis-platform.html)
- Intel Distiller [Link](https://github.com/NervanaSystems/distiller)
  - Compression Library ontop of PyTorch
  - State of the art algorithms
  - Includes Pruning, Quantisation, Regularization, Knowledge Distilation, Conditional Computation
- Keras-Surgeon [Link](https://github.com/BenWhetton/keras-surgeon)
- QNNPACK [Link](https://github.com/pytorch/QNNPACK)
- MXNet Contrib Quantization [Link](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Graph+Optimization+and+Quantization+based+on+subgraph+and+MKL-DNN)
- tensorflow_model_optimization [Link](https://www.tensorflow.org/model_optimization)
- Chainer Pruner [link](https://github.com/DeNA/ChainerPruner)

Deployment Tools:

- TVM Neural Network Compiler Stack
  - Reay
  - VTA
- Glow
- OpenVino
  - Deployment and model optimization for FPGA
