# Digit Recognizer

This repository contains a project that uses a neural network to recognize handwritten digits from the MNIST dataset, obtained from the Kaggle Digit Recognizer competition. The model is implemented using a simple Artificial Neural Network (ANN) and trained to classify digits from 0 to 9.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview
This project implements a digit recognizer using an Artificial Neural Network (ANN). The model is trained on the MNIST dataset, which contains 28x28 pixel grayscale images of handwritten digits. The network learns to classify these digits with the goal of achieving high accuracy.

## Dataset
The dataset used for this project is obtained from the Kaggle [Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer/data). It consists of:
- **Train Data:** 42,000 samples with labeled images
- **Test Data:** 28,000 samples without labels (for competition submission)
- **Image Size:** 28x28 pixels (flattened to 784 input features)

You can download the dataset from the competition page:
- [Kaggle Digit Recognizer Dataset](https://www.kaggle.com/c/digit-recognizer/data)

## Model Architecture
The model is a simple feedforward neural network with the following layers:
- **Input Layer:** 784 neurons (28x28 input size)
- **Hidden Layer:** 10 neurons with ReLU activation
- **Output Layer:** 10 neurons with softmax activation for multi-class classification

## Requirements
To run this project, you need the following dependencies installed:
- `numpy`
- `matplotlib`

You can install the required libraries using pip:

```bash
pip install numpy matplotlib

