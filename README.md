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

The Convolutional Neural Network (CNN) architecture has been designed and optimized for depth estimation tasks. I utilize several convolutional layers, pooling layers, and a regression output layer to predict the depth of each pixel.

## Installation

1. Clone this repository: 
```
git clone https://github.com/wiadarola/CV-Monocular-Depth-Estimator.git
```

2. Navigate to the directory:
```
cd CV-Monocular-Depth-Estimator
```

## Usage

1. Run the model:
```
python project.py
```

## Results

The trained CNN shows promising results, with tests demonstrating close approximations to actual depth camera outputs.
