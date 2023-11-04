import numpy as np


class Convolution:
    def __init__(self, kernel_num, kernel_depth, kernel_size):
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernel_depth = kernel_depth

        # Generate random filters of shape (kernel_num, kernel_size, kernel_size, colors).
        self.kernels = np.random.randn(
            kernel_num,
            kernel_depth,
            kernel_size,
            kernel_size,
        ) / (kernel_size**2)

    def get_forward_prop_output_shape(self):
        original_h, original_w, original_c = self.original.shape

        # Initialize the convolution output volume of the correct size
        output_h = original_h - self.kernel_size + 1
        output_w = original_w - self.kernel_size + 1

        return output_h, output_w, original_c

    def forward(self, input):
        # Save for backward propagation
        self.original = input
        output_h, output_w, image_c = self.get_forward_prop_output_shape()

        # Shape the output
        convolution_output = np.zeros(
            (self.kernel_num,
             output_h,
             output_w)
        )

        for k in range(self.kernel_num):
            for c in range(image_c):
                for h in range(output_h):
                    for w in range(output_w):
                        # Create a region of the kernel size in the input from each cell of the output
                        region = self.original[
                            h:(h + self.kernel_size),
                            w:(w + self.kernel_size),
                            c
                        ]

                        convolution_output[k, h, w] += np.sum(
                            region * self.kernels[k, c, :, :]
                        )

        return convolution_output

    def backward(self, gradients, learning_rate):
        # Shape of the kernels that needs to be updated
        dE_dk = np.zeros(self.kernels.shape)

        # Dimensions of the input of this method
        output_h, output_w, image_c = self.get_forward_prop_output_shape()

        for kernel in range(self.kernel_num):
            for c in range(image_c):
                for h in range(output_h):
                    for w in range(output_w):
                        # Create a region in the input from each cell of the output
                        region = self.original[
                            h:(h + self.kernel_size),
                            w:(w + self.kernel_size),
                            c
                        ]

                        dE_dk[kernel, c, :, :] += region * \
                            gradients[kernel, h, w]

        # Update the parameters
        self.kernels -= learning_rate*dE_dk

        return dE_dk
