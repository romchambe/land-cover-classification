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
        ) / (kernel_size**2)

    def get_output_shape_from_image(self):
        image_h, image_w, image_c = self.image.shape

        # Initialize the convolution output volume of the correct size
        output_h = image_h - self.kernel_size + 1
        output_w = image_w - self.kernel_size + 1

        return output_h, output_w, image_c

    def forward(self, image):
        # Save for backward propagation
        self.image = image

        # Initialize the convolution output volume of the correct size
        output_h, output_w, image_c = self.get_output_shape_from_image()

        # Shape the output
        convolution_output = np.zeros(
            (output_h,
             output_w,
             image_c,
             self.kernel_num)
        )

        for h in range(output_h):
            for w in range(output_w):
                for c in range(image_c):
                    for kernel in self.kernels:
                        # Create a region in the input from each cell of the output
                        region = self.image[
                            h:(h + self.kernel_size),
                            w:(w + self.kernel_size),
                            c
                        ]

                        convolution_output[h, w, c] = np.sum(
                            region * kernel[:, :, c]
                        )

        return convolution_output

    def backward(self, dE_dY, alpha):

        output_h, output_w, image_c = self.get_output_shape_from_image()
        # Initialize gradients according to the shape of kernels
        dE_dk = np.zeros(self.kernels.shape)

        for h in range(output_h):
            for w in range(output_w):
                for c in range(image_c):
                    # Create a region in the input from each cell of the output
                    region = self.image[
                        h:(h + self.kernel_size),
                        w:(w + self.kernel_size),
                        c
                    ]

                    for kernel in range(self.kernel_num):
                        dE_dk[kernel, :, :, c] += region * \
                            dE_dY[h, w, c, kernel]

        # Update the parameters
        self.kernels -= alpha*dE_dk

        return dE_dk
