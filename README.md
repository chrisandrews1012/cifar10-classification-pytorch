# PyTorch CNNs for CIFAR-10 Image Classification

This repository contains two Jupyter notebooks from my graduate-level course project in Software-Hardware Co-design for Intelligent Systems. The goal was to implement, train, and analyze Convolutional Neural Networks (CNNs) for image classification on the CIFAR-10 dataset using PyTorch.

## Notebooks

### 1. `lenet5_cifar10_pytorch.ipynb`

This notebook provides a comprehensive walkthrough of building the classic LeNet-5 architecture from the ground up.

**Key Activities:**
*   **Theoretical Analysis:** Manual calculation of the model's memory footprint (parameters) and computational cost (MACs).
*   **PyTorch Implementation:** Defines the `LeNet5` model class, including optional batch normalization layers.
*   **Data Pipeline:** Implements a full data preprocessing and augmentation pipeline using `torchvision.transforms`.
*   **Training & Evaluation:** Contains the complete training and validation loop, including hyperparameter tuning, L2 regularization, and a learning rate decay schedule to optimize performance.

### 2. `resnet_cifar10_pytorch.ipynb`

Building upon the first part of the lab, this notebook implements a more advanced custom CNN with residual connections (similar to a ResNet) to achieve higher classification accuracy on CIFAR-10.

**Key Activities:**
*   **Advanced Architecture:** Defines a custom `CustomDNN` class incorporating residual blocks to improve gradient flow and enable deeper training.
*   **Performance Tuning:** Focuses on hyperparameter optimization to push validation accuracy significantly beyond the LeNet-5 baseline.
*   **Inference:** Includes logic for running inference on the test set and generating a submission file, simulating a competitive or Kaggle-style task.

## Note on Project Context

This project was completed as part of a university lab assignment. To uphold academic integrity and prevent the direct reuse of assignment starter code, I excluded the helper utility files from this repository. 

The core logic for the model implementation, data processing, and training is fully contained within the notebooks.
