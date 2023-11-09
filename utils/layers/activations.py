import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return x * (x > 0)


def relu_prime(x):
    return 1. * (x > 0)


class Sigmoid():
    def forward(self, input):
        self.input = input
        return sigmoid(self.input)

    def backward(self, gradients, learning_rate):
        return np.multiply(gradients, sigmoid_prime(self.input))


class Relu():
    def forward(self, input):
        self.input = input
        return relu(input)

    def backward(self, gradients, learning_rate):
        return np.multiply(gradients, relu_prime(self.input))


class HyperbolicTan():
    def forward(self, input):
        self.input = input
        return np.tanh(input)

    def backward(self, gradients, learning_rate):
        return np.multiply(gradients, 1 - np.tanh(self.input) ** 2)
