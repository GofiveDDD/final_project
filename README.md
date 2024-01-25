# Fashion MNIST Classification using CNN


This repository contains a simple Convolutional Neural Network (CNN) model for classifying images from the Fashion MNIST dataset. The model is implemented in Python using TensorFlow and Keras.

## Overview

Fashion MNIST is a dataset of Zalando's article images, consisting of 60,000 28x28 grayscale images of 10 fashion categories. The goal is to train a CNN model to accurately classify these images.

## Getting Started

### Prerequisites

Make sure you have the following dependencies installed:

- Python
- TensorFlow
- Keras

You can install them using the following command:

```bash
pip install tensorflow keras

#### Running the Code
Clone this repository and run the fashion_mnist_cnn.py script to train and evaluate the CNN model:
python fashion_mnist_cnn.py

##### Training
The model is trained using the Fashion MNIST training dataset. The training parameters such as learning rate, batch size, and dropout rate can be adjusted in the script.

###### Evaluation
The model is evaluated on a validation set to assess its performance. The evaluation results, including loss and accuracy, are printed at the end of the training process.

#######  Model
The trained model is saved to the model_checkpoint directory. You can load this model for further evaluation or use in other applications.
loaded_model = tf.keras.models.load_model('model_checkpoint')

####### Author
