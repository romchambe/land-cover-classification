import numpy as np
from datetime import datetime
from .preprocess import INTERMEDIARY_FILE_PATH

from .layers.convolution import Convolution
from .layers.max_pooling import MaxPooling
from .layers.soft_max import Softmax
from .layers.reshape import Reshape
from .layers.dense import Dense
from .layers.activations import HyperbolicTan

layers = [
    Convolution(12, 3, 3),  # 62*62*12
    HyperbolicTan(),
    MaxPooling(3),  # 20*20*12
    Reshape((12, 20, 20), (20*20*12, 1)),
    Dense(20*20*12, 10),
    Softmax()
]


def propagate_forward(image, label, layers):
    output = image
    for layer in layers:
        output = layer.forward(output)

    # Compute loss (cross-entropy) and accuracy
    loss = -np.log(output[label])

    accuracy = 1 if np.argmax(output) == label else 0
    return output, loss, accuracy


def propagate_backward(gradient, layers, learning_rate):
    grad_back = gradient

    for layer in layers[::-1]:
        grad_back = layer.backward(grad_back, learning_rate)

    return grad_back


def train_on_image(image, label, layers, learning_rate):
    # Forward step
    prediction, loss, accuracy = propagate_forward(image, label, layers)

    # Gradients of loss function (cross-entropy)
    loss_gradients = np.zeros(10)
    loss_gradients[label] = -1 / prediction[label]

    # Backprop step
    propagate_backward(loss_gradients, layers, learning_rate)

    return loss, accuracy


def train():
    npzfile = np.load(INTERMEDIARY_FILE_PATH)

    x_train = npzfile['x_train']
    y_train = npzfile['y_train']

    for epoch in range(15):
        # Shuffle data and labels using the same permutation and slice the first 5000 results
        permutation = np.random.permutation(len(x_train))

        x_train = np.array(x_train)[permutation][:1000]
        y_train = np.array(y_train)[permutation][:1000]

        accuracy, loss = 0, 0
        beginning = datetime.utcnow()

        print("=============================================")
        print(f"Epoch #{epoch + 1}")

        for i in range(len(x_train)):
            image = x_train[i]
            label = y_train[i]

            loss_on_image, accurate = train_on_image(
                image,
                label,
                layers,
                0.02
            )

            loss += loss_on_image
            accuracy += accurate

            if (i+1) % 100 == 0:
                print(
                    f"Iteration {i+1} | Avg loss on epoch {loss / (i+1)}")

        epoch_duration = datetime.utcnow() - beginning

        print("=============================================")
        print(f"Achieved accuracy : {accuracy / 10} %")
        print(f"Average loss : {loss / 1000}")
        print(f"Epoch duration : {str(epoch_duration)}")
        print("=============================================")
