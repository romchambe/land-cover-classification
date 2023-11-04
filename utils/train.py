import numpy as np
from .preprocess import INTERMEDIARY_FILE_PATH

from .layers.convolution import Convolution
from .layers.max_pooling import MaxPooling
from .layers.soft_max import Softmax
from .layers.reshape import Reshape

layers = [
    Convolution(12, 3, 3),  # 62*62*12
    MaxPooling(3),  # 20*20*12
    Reshape(),
    Softmax(20*20*12, 10)
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
    output, loss, accuracy = propagate_forward(image, label, layers)

    # Initiate gradients with the derivative of the loss
    gradients = np.zeros(10)
    gradients[label] = -1 / output[label]

    # Backprop step
    gradient_back = propagate_backward(gradients, layers, learning_rate)

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

        print(f"Epoch #{epoch + 1} ----- ")

        for i in range(len(x_train)):
            if i % 100 == 0:
                print(f"""Image #{i}. Over the past 100 images : 
                - Average loss : {loss / 10}
                - Accuracy : {accuracy} / 10
              """)

                loss, accuracy = 0, 0

            image = x_train[i]
            label = y_train[i]

            loss_on_image, accurate = train_on_image(
                image,
                label,
                layers,
                0.015
            )

            loss += loss_on_image
            accuracy += accurate

        print(f"Achieved accuracy {accuracy / 10} %")
        print(f"Average loss {loss / 1000}")
