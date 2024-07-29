# VGG Network Implementation

This repository contains the implementation of the VGG network, based on the seminal paper "Very Deep Convolutional Networks for Large-Scale Image Recognition". The implementation includes training and evaluation on the Tiny ImageNet dataset, with support for various training and evaluation techniques.

## Features
- **Single Scale Training**: Train the VGG network with images of a consistent size.
- **Multi-Scale Training**: Train the VGG network with images of varying scales to enhance robustness.
- **Single Scale Evaluation**: Evaluate the model using single scale images.
- **Multi-Scale Evaluation**: Evaluate the model using images of multiple scales and aggregate predictions.
- **Multi-Crop Evaluation**: Generate multiple crops of input images and average the predictions for improved accuracy.
- **Metrics Calculation**: Calculate and report Top-1 error, Top-5 error, and accuracy.

## Tech Stack
- Python
- PyTorch
- NumPy

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vgg-network-implementation.git
   cd vgg-network-implementation
   ```
