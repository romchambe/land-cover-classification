import numpy as np


class ConvolutionLayer:
    def __init__(self, kernel_num, kernel_size):
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        # Generate random filters of shape (kernel_num, kernel_size, kernel_size, colors).
        self.kernels = np.random.randn(
            kernel_num,
            kernel_size,
            kernel_size,
            3
        )

    def forward(self, image):
        # Extract image height and width
        image_h, image_w, image_c = image.shape

        # Initialize the convolution output volume of the correct size
        output_h = image_h - self.kernel_size + 1
        output_w = image_w - self.kernel_size + 1

        # Shape the output
        convolution_output = np.zeros(
            (output_h,
             output_w,
             image_c,
             self.kernel_num)
        )

        for i in range(output_h):
            for j in range(output_w):
                for c in range(3):  # Iterate over the RGB channels
                    convolution_output[i, j, c] = np.sum(
                        image[i:i+self.kernel_size, j:j+self.kernel_size, c] * self.kernels[:, :, :, c])

        return convolution_output
