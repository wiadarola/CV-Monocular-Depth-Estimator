# 2D to Depth Image Estimation

Utilizing Convolutional Neural Networks (CNN) to transform 2D RGB images into depth images.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Overview

This project aims to estimate depth images from 2D RGB images. By training a CNN with 2D images as input and depth camera images as the target, I provide a model that can predict depth for each pixel of a standard image, offering a depth perception that can be critical for applications such as robotics, augmented reality, and computer vision tasks.

## Model Architecture

The Convolutional Neural Network (CNN) architecture has been designed and optimized for depth estimation tasks. We utilize several convolutional layers, pooling layers, and a regression output layer to predict the depth of each pixel.

## Installation

1. Clone this repository: 
```
git clone https://github.com/wiadarola/2d-to-depth-estimation.git
```

2. Navigate to the directory:
```
cd 2d-to-depth-estimation
```

3. Install the required libraries:
```
pip install -r requirements.txt
```

## Usage

1. Train the model:
```
python train.py
```

2. Predict depth image from a 2D image:
```
python predict.py --input [path_to_input_image]
```

## Results

The trained CNN shows promising results, with our tests demonstrating close approximations to actual depth camera outputs.
