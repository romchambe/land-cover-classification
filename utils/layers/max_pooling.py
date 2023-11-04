import numpy as np


class MaxPooling:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def get_forward_prop_output_shape(self):
        num_kernels, input_h, input_w = self.original.shape

        output_h = input_h // self.kernel_size
        output_w = input_w // self.kernel_size

        return num_kernels, output_h, output_w

    def forward(self, input):
        # Save for backward propagation
        self.original = input

        num_kernels, output_h, output_w = self.get_forward_prop_output_shape()

        # Shape the output
        max_pooling_output = np.zeros(
            (num_kernels, output_h, output_w)
        )

        # Iterate over the output matrix for each RGB color
        for h in range(output_h):
            for w in range(output_w):
                # Create a region from the input based on the output cell coordinate
                region = input[
                    :,
                    (h * self.kernel_size):(h * self.kernel_size + self.kernel_size),
                    (w * self.kernel_size):(w * self.kernel_size + self.kernel_size),
                ]

                # We assign the max value for the region to the output along height, width and color axis
                max_pooling_output[:, h, w] = np.amax(
                    region,
                    axis=(1, 2)
                )

        return max_pooling_output

    def backward(self, gradients, learning_rate):
        # Shape output of the method
        # Original input shape - output of the backward propagation
        dE_dk = np.zeros(self.original.shape)

        # Dimensions of the input of this method
        input_h, input_w, num_kernels = self.get_forward_prop_output_shape()

        for h in range(input_h):
            for w in range(input_w):
                # Create a region from each cell from the input
                region = self.original[
                    :,
                    (h * self.kernel_size):(h * self.kernel_size + self.kernel_size),
                    (w * self.kernel_size):(w * self.kernel_size + self.kernel_size),
                ]

                num_kernels, region_height, region_width = region.shape
                max_of_region = np.amax(region, axis=(1, 2))

                for kernel in range(num_kernels):
                    for region_h in range(region_height):
                        for region_w in range(region_width):
                            if region[kernel, region_h, region_w] == max_of_region[kernel]:
                                # We just pass the gradients. Max pooling is just a compression step
                                dE_dk[
                                    kernel,
                                    h * self.kernel_size + region_h,
                                    w * self.kernel_size + region_w,
                                ] = gradients[kernel, h, w]

        return dE_dk
