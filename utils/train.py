import numpy as np
from .preprocess import INTERMEDIARY_FILE_PATH

from .layers.convolution import ConvolutionLayer
from .layers.max_pooling import MaxPoolingLayer
from .layers.soft_max import SoftmaxLayer

layers = [
    ConvolutionLayer(12, 5),  # 60x60x3x12
    MaxPoolingLayer(6),  # 10x10x3x12
    SoftmaxLayer(10*10*3*12, 10)
]


def propagate_forward(image, label, layers):
    output = image
    for layer in layers:
        output = layer.forward(output)

    # Compute loss (cross-entropy) and accuracy
    loss = -np.log(output[label])

    accuracy = 1 if np.argmax(output) == label else 0
    return output, loss, accuracy


def propagate_backward(gradient, layers, alpha=0.05):
    grad_back = gradient

    for layer in layers[::-1]:
        if type(layer) in [ConvolutionLayer, SoftmaxLayer]:
            grad_back = layer.backward(grad_back, alpha)
            # print(type(layer), grad_back)
        elif type(layer) == MaxPoolingLayer:
            grad_back = layer.backward(grad_back)
            # print(type(layer), grad_back)

    return grad_back


def train_on_image(image, label, layers, alpha=0.05):
    # Forward step
    output, loss, accuracy = propagate_forward(image, label, layers)

    # Initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1/output[label]

    # Backprop step
    propagate_backward(gradient, layers, alpha)

    return loss, accuracy


def train():
    npzfile = np.load(INTERMEDIARY_FILE_PATH)

    x_train = npzfile['x_train']
    y_train = npzfile['y_train']

    # Shuffle data and labels using the same permutation and slice the first 5000 results
    permutation = np.random.permutation(len(x_train))

    x_train = np.array(x_train)[permutation][:5000]
    y_train = np.array(y_train)[permutation][:5000]

    accuracy, loss = 0, 0
    for i in range(len(x_train)):

        if i % 100 == 0:
            print(f"""Image #{i}. Over the past 100 images : 
              - Average loss : {loss / 100}
              - Accuracy : {accuracy} %
            """)

            loss, accuracy = 0, 0

        image = x_train[i]
        label = y_train[i]
        # if (i < 2):
        loss_on_image, accurate = train_on_image(image, label, layers)
        loss += loss_on_image
        accuracy += accurate
        # else:
        #     break
