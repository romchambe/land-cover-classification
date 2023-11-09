import numpy as np


class Dense:
    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.randn(output_size, input_size) / input_size
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.original = input
        return self.weights @ input + self.bias

    def backward(self, gradients, learning_rate):
        # Gradient of loss with respect to weight, bias, input
        weight_gradients = np.dot(gradients, self.original.T)
        bias_gradients = gradients
        gradients_rel_to_input = np.dot(self.weights.T, gradients)

        self.weights -= learning_rate * weight_gradients
        self.bias -= learning_rate * bias_gradients

        return gradients_rel_to_input
