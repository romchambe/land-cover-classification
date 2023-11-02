import numpy as np


class MaxPoolingLayer:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward(self, image):
        image_h, image_w, image_c, num_kernels = image.shape

        output_h = image_h // self.kernel_size
        output_w = image_w // self.kernel_size

        # Shape the output
        max_pooling_output = np.zeros(
            (output_h, output_w, 3, num_kernels)
        )

        for h in range(output_h):
            for w in range(output_w):
                for c in range(3):
                    max_pooling_output[h, w] = np.amax(
                        image[
                            (h*self.kernel_size):(h*self.kernel_size+self.kernel_size),
                            (w*self.kernel_size):(w*self.kernel_size+self.kernel_size),
                            c
                        ],
                        axis=(0, 1, 2)
                    )

        return max_pooling_output
