import numpy as np


class SoftmaxLayer:
    def __init__(self, input_units, output_units):
        # Initiallize weights and biases
        self.weight = np.random.randn(input_units, output_units)/input_units
        self.bias = np.zeros(output_units)

    def forward(self, image):
        self.original_shape = image.shape

        # Flatten the image
        image_flattened = image.flatten()

        # Save it for backward propagation
        self.flattened = image_flattened

        # Perform matrix multiplication and add bias
        first_output = np.dot(image_flattened, self.weight) + self.bias
        self.output = first_output

        # Apply softmax activation
        softmax_output = np.exp(first_output) / \
            np.sum(np.exp(first_output), axis=0)

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
            dZ_dw = self.flattened
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

            return dE_dX.reshape(self.original_shape)
