import numpy as np
from .preprocess import INTERMEDIARY_FILE_PATH

from .layers.convolution import ConvolutionLayer
from .layers.max_pooling import MaxPoolingLayer
from .layers.soft_max import SoftmaxLayer

layers = [
    ConvolutionLayer(10, 5),  # 60x60x3x12
    MaxPoolingLayer(3),  # 20x20x3x12
    SoftmaxLayer(20*20*3*10, 10)
]


def propagate_forward(image, label, layers):
    output = image
    for layer in layers:
        output = layer.forward_prop(output)

    # Compute loss (cross-entropy) and accuracy
    loss = -np.log(output[label])
    accuracy = 1 if np.argmax(output) == label else 0
    return output, loss, accuracy


def train_on_image(image, label, layers):
    # Forward step
    output, loss, accuracy = propagate_forward(image, label, layers)
    print(output)
    return loss, accuracy


def train():
    npzfile = np.load(INTERMEDIARY_FILE_PATH)

    x_train = npzfile['x_train']
    y_train = npzfile['y_train']

    for i in range(len(x_train)):
        image = x_train[i]
        label = y_train[i]

        train_on_image(image, label, layers)
