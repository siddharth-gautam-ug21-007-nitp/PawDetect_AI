# PawDetect AI

PawDetect AI is a Convolutional Neural Network (CNN) model designed to classify images of cats and dogs. The model is built using TensorFlow and Keras, and it is trained on the popular **Dogs vs Cats** dataset from Kaggle. This project demonstrates the use of deep learning for binary image classification.

## Overview

This project aims to accurately predict whether an image contains a dog or a cat. It leverages a CNN architecture with multiple convolutional, pooling, and dense layers. The model is trained using normalized image datasets, with overfitting prevention techniques like Batch Normalization and Dropout.

## Features

- **Dataset**: download and processes the **Dogs vs Cats** dataset from Kaggle.
- **Model Architecture**: A CNN with:
  - Convolutional layers for feature extraction.
  - MaxPooling layers for dimensionality reduction.
  - BatchNormalization and Dropout for reducing overfitting.
- **Evaluation**: The model is evaluated using training and validation accuracy and loss metrics.
- **Prediction**: It can predict unseen images as either a dog or a cat.

## Dataset

The **Dogs vs Cats** dataset is sourced from Kaggle. The dataset consists of two folders:
- **train**: Contains the labeled images for training.
- **test**: Contains the labeled images for validation.

## Model Architecture

The CNN model consists of multiple layers designed to extract features and classify images:
- Convolutional layers with filters for identifying patterns in the images.
- MaxPooling layers to reduce the dimensions of the data.
- BatchNormalization to normalize the activations and improve training.
- Dropout layers to prevent overfitting and improve generalization.
- Dense layers at the end for classification, with a Sigmoid function to output binary results (cat or dog).

## Results

The model can accurately classify new, unseen images as either a dog or a cat based on the trained data. The accuracy and loss of the model are tracked throughout the training process to ensure optimal performance.
