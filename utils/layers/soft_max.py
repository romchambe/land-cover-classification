import numpy as np


class Softmax:
    def __init__(self, input_size, output_size):
        # Initiallize weights and biases
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.bias = np.zeros(output_size)

    def forward(self, input):
        self.original = input

        # Densify the output using weights and biases
        dense_output = np.dot(input, self.weights) + self.bias
        self.output = dense_output

        # Apply softmax activation
        softmax_output = np.exp(dense_output) / \
            np.sum(np.exp(dense_output), axis=0)

        return softmax_output

    def backward(self, gradients, alpha):
        for i, loss_gradient in enumerate(gradients):
            if loss_gradient == 0:
                continue

            transformation_eq = np.exp(self.output)
            S_total = np.sum(transformation_eq)

            # Compute gradients with respect to output (Z)
            softmax_gradients = - \
                transformation_eq[i] * transformation_eq / (S_total**2)

            softmax_gradients[i] = transformation_eq[i] * \
                (S_total - transformation_eq[i]) / (S_total**2)

            # Gradient of loss with respect of output
            softmax_output_gradients = loss_gradient * softmax_gradients

            # Gradient of loss with respect to weight, bias, input
            weight_gradients = self.original[np.newaxis].T @ softmax_output_gradients[np.newaxis]
            bias_gradients = softmax_output_gradients
            output_gradients = self.weights @ softmax_output_gradients

            # Update weights and bias for the densification
            self.weights -= alpha*weight_gradients
            self.bias -= alpha*bias_gradients

            return output_gradients
