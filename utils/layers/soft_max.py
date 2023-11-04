import numpy as np


class Softmax:
    def __init__(self, input_size, output_size):
        # Initiallize weights and biases
        self.weight = np.random.randn(input_size, output_size)/input_size
        self.bias = np.zeros(output_size)

    def forward(self, input):
        self.original = input

        # Densify
        dense_output = np.dot(input, self.weight) + self.bias
        self.output = dense_output

        # Apply softmax activation
        softmax_output = np.exp(dense_output) / \
            np.sum(np.exp(dense_output), axis=0)

        return softmax_output

    def backward(self, dE_dY, alpha):
        for i, gradient in enumerate(dE_dY):
            if gradient == 0:
                continue

            transformation_eq = np.exp(self.output)
            S_total = np.sum(transformation_eq)

            # Compute gradients with respect to output (Z)
            dY_dZ = -transformation_eq[i]*transformation_eq / (S_total**2)
            dY_dZ[i] = transformation_eq[i] * \
                (S_total - transformation_eq[i]) / (S_total**2)

            # Compute gradients of output Z with respect to weight w, bias b, input
            dZ_dw = self.original
            dZ_db = 1
            dZ_dX = self.weight

            # Gradient of loss with respect ot output
            dE_dZ = gradient * dY_dZ

            # Gradient of loss with respect to weight, bias, input
            dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis]
            dE_db = dE_dZ * dZ_db
            dE_dX = dZ_dX @ dE_dZ

            # Update parameters
            self.weight -= alpha*dE_dw
            self.bias -= alpha*dE_db

            return dE_dX
