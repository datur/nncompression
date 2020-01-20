# A Survey of Model Compression and Acceleration for Deep Neural Networks

# Summary

- This paper reviews recent work in on compressing and accelerating deep neural networks for deployment on edge devices.
- The approaces in this paper can be divided into 4 categories:

  - Parameter prining and sharing

    - explores the redundancy in the model parameters with the aim to remove parameters evaluated to be redundant or uncritical.
    - Takes the form of 3 categories in this paper:
      - Quantization and binarization
        - Compressed the original network model by reducing the bits required to represent each weight.
        - References of papers using constrained bit width weights
        - Y. Gong, L. Liu, M. Yang, and L. D. Bourdev, “Compressingdeep convolutional networks using vector quantization,” CoRR, vol. abs/1412.6115, 2014.
        - Y. W. Q. H. Jiaxiang Wu, Cong Leng and J. Cheng, “Quantized convolutional neural networks for mobile devices,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
        - V. Vanhoucke, A. Senior, and M. Z. Mao, “Improving the speed of neural networks on cpus,” in Deep Learning and Unsupervised Feature Learning Workshop, NIPS 2011, 2011.
        - S. Gupta, A. Agrawal, K. Gopalakrishnan, and P. Narayanan, “Deep learning with limited numerical precision,” in Proceedings of the 32Nd International Conference on International Conference on Machine Learning - Volume 37, ser. ICML’15, 2015, pp. 1737–1746.
        - S. Han, H. Mao, and W. J. Dally, “Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding,” International Conference on Learning Representations (ICLR), 2016.
        - Binary networks
          - Extreme case compression
          - BinaryConnect, BinaryNet and XNORNetworks
        - Drawbacks
          - Accuracy of binary networks significantly lowered when dealing with large CNN Architectures
      - Pruning and sharing
        - Network pruning and sharing has been used both to reduce network complexity and to address the over-fitting issue
          - Biased Weight Decay
          - Optimal Brain Damage
          - Optimal Brain Surgeon
        - OBD and OBS reduced the number of connections based on the Hessian of the loss function, and their work suggested that such pruning gave higher accuracy than magnitudebased pruning such as the weight decay method.
        - Lately the direction is to prune redundant, non-informative weights in a pre-trained CNN model
          - S. Srinivas and R. V. Babu, “Data-free parameter pruning for deep neural networks,” in Proceedings of the British Machine Vision Conference 2015, BMVC 2015, Swansea, UK, September 7-10, 2015, 2015, pp. 31.1–31.12.
          - S. Han, J. Pool, J. Tran, and W. J. Dally, “Learning both weights and connections for efficient neural networks,” in Proceedings of the 28th International Conference on Neural Information Processing Systems, ser. NIPS’15, 2015.
          - W. Chen, J. Wilson, S. Tyree, K. Q. Weinberger, and Y. Chen, “Compressing neural networks with the hashing trick.” JMLR Workshop and Conference Proceedings, 2015.
          - K. Ullrich, E. Meeds, and M. Welling, “Soft weight-sharing for neural network compression,” CoRR, vol. abs/1702.04008, 2017
        - There is also growing interest in training compact CNNs with sparsity constraints. Those sparsity constraints are typically introduced in the optimization problem as l0 or l1- norm regularizers
          - V. Lebedev and V. S. Lempitsky, “Fast convnets using group-wise brain damage,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, 2016, pp. 2554–2564.
          - H. Zhou, J. M. Alvarez, and F. Porikli, “Less is more: Towards compact cnns,” in European Conference on Computer Vision, Amsterdam, the Netherlands, October 2016, pp. 662–677.
          - W. Wen, C. Wu, Y. Wang, Y. Chen, and H. Li, “Learning structured sparsity in deep neural networks,” in Advances in Neural Information Processing Systems 29, D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, Eds., 2016, pp. 2074–2082.
          - H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf, “Pruning filters for efficient convnets,” CoRR, vol. abs/1608.08710, 2016.
        - Drawbacks
          - First, pruning with l1 or l2 regularization requires more iterations to converge than general. In addition, all pruning criteria require manual setup of sensitivity for layers, which demands fine-tuning of the parameters and could be cumbersome for some applications.
      - Designing Structural Matricies
        - the structure should not only reduce the memory cost, but also dramatically accelerate the inference and training stage via fast matrix-vector multiplication and gradient computations.
        - An Example of this is a circulant matrix
        -

* low-rank factorization
  - uses matrix/tensor decomposition to estimate the informative parameters
* transferred/compact convolutional filters
  - design special structural convolutional filters to reduce the parameter space
* knowledge distilation

  - learn a problem on a larger network and use this to train a smaller network

* Low rank factorization and transferred/compact filters are end to end solutions
* however weight sharing and pruning use different methods making the process more complex. Using:
* vector quantization, binary coding, and sparse constraints. This usually takes several steps.
*

# Quotes

- deep networks with millions or even billions of parameters, and the availability of GPUs with very high computation capability plays a key role in their success
- As larger neural networks with more layers and nodes are considered, reducing their storage and computational cost becomes critical
- fundamental challenges in deploying deep learning systems to portable devices with limited resources (e.g. memory, CPU, energy, bandwidth)
- Resnet 50 needs 95MB to store model with 3.8 billion floating point operations
-
-
