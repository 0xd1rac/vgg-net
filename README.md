# VGG Network Implementation

This repository contains the implementation of the VGG network, based on the seminal paper "Very Deep Convolutional Networks for Large-Scale Image Recognition". The implementation includes training and evaluation on the Tiny ImageNet dataset, with support for various training and evaluation techniques.

## Dataset
[link text] (https://huggingface.co/datasets/zh-plus/tiny-imagenet)

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
   git clone git@github.com:0xd1rac/vgg-net.git
   cd vgg-net
   ```

2. Install the required packages
  ```bash
   pip install -r requirements.txt
   ```
## Training 
### Single Scale Training
   ```bash

   ```

### Multi-Scale Training 
   ```bash

   ```

## Evaluation
### Single Scale Training 
   ```bash

   ```

### Multi-Scale Training
   ```bash

   ```

### Multi-Crop Training
   ```bash

   ```


