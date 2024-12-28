# D2N-Dual-Path-Attention-Network-Using-Domain-Adaption-Strategy-For-Medical-Image-Segmentation
This repository contains the implementation of the D2N: A Dual Path Architecture based on UNet , as described in our paper. The model is designed for high-performance medical image segmentation tasks, such as brain tumor, skin lesion, and polyp segmentation. Furthermore, this code can be applied to wide range of meical image segmentation datasets.

Instructions: Clone the repository: git clone https://github.com/nooriahmed/D2N-Dual-Path-Attention-Network-Using-Domain-Adaption-Strategy-For-Medical-Image-Segmentation

Key instructions:

If find the issues of overfitting during training then restart training with a small batch size. Please follow the same pattern of folder settings of datasets as used in our code. check layers shapes if resize images has been done. Please make sure to avoid any class imbalance inside the mentioned datasets, in such case loss function could be in negative during training. In case of any uncertainity of classes or labelling in datasets or image color occlusions. A small difference in results may occur. Follow the instructions in README.md to train and evaluate the model

Requirements: Python vesion 3.10.12 TensorFlow version 2.17.0 Keras 3.4.1
