import numpy as np


class Softmax:

    def forward(self, input):
        self.original = input

        return (np.exp(input) / np.sum(np.exp(input), axis=0)).flatten()

    def backward(self, gradients, alpha):
        for i, loss_gradient in enumerate(gradients):
            if loss_gradient == 0:
                continue

            transformation_eq = np.exp(self.original)
            S_total = np.sum(transformation_eq)

            # Compute gradients with respect to output (Z)
            softmax_gradients = - \
                transformation_eq[i] * transformation_eq / (S_total**2)

            softmax_gradients[i] = transformation_eq[i] * \
                (S_total - transformation_eq[i]) / (S_total**2)

            # Gradient of loss with respect of output
            return loss_gradient * softmax_gradients
