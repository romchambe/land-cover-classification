import numpy as np


class MaxPoolingLayer:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def get_output_shape_from_image(self):

        image_h, image_w, image_c, num_kernels = self.image.shape

        output_h = image_h // self.kernel_size
        output_w = image_w // self.kernel_size

        return (output_h, output_w)

    def forward(self, image):
        # Save for backward propagation
        self.image = image
        image_h, image_w, image_c, num_kernels = image.shape

        output_h, output_w = self.get_output_shape_from_image()

        # Shape the output
        max_pooling_output = np.zeros(
            (output_h, output_w, 3, num_kernels)
        )

        # Iterate over the output matrix for each RGB color
        for h in range(output_h):
            for w in range(output_w):
                for c in range(image_c):
                    # Create a region from the input based on the output cell coordinate
                    region = image[
                        (h * self.kernel_size):(h * self.kernel_size + self.kernel_size),
                        (w * self.kernel_size):(w * self.kernel_size + self.kernel_size),
                        c
                    ]

                    # We assign the max value for the region to the output along height, width and color axis
                    max_pooling_output[h, w, c] = np.amax(
                        region,
                        axis=(0, 1)
                    )

        return max_pooling_output

    def backward(self, dE_dY):
        dE_dk = np.zeros(self.image.shape)
        image_h, image_w, image_c, num_kernels = self.image.shape

        # These are the dimensions of our input in the case of forward propagation
        input_h, input_w = self.get_output_shape_from_image()

        for h in range(input_h):
            for w in range(input_w):
                for c in range(image_c):
                    # Create a region from each cell from the input
                    region = self.image[
                        (h * self.kernel_size):(h * self.kernel_size + self.kernel_size),
                        (w * self.kernel_size):(w * self.kernel_size + self.kernel_size),
                        c
                    ]

                    region_height, region_width, num_kernels = region.shape
                    max_of_region = np.amax(region, axis=(0, 1))

                    for region_h in range(region_height):
                        for region_w in range(region_width):
                            for kernel in range(num_kernels):
                                # We fill the cells corresponding to the max of the region
                                # with the gradient coming from the previous layer
                                if region[region_h, region_w, kernel] == max_of_region[kernel]:
                                    # We just pass the gradients. Max pooling is just a compression step
                                    dE_dk[
                                        h * self.kernel_size + region_h,
                                        w * self.kernel_size + region_w,
                                        c,
                                        kernel
                                    ] = dE_dY[h, w, c, kernel]

        return dE_dk
