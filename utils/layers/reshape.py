class Reshape:
    def forward(self, input):
        self.original_shape = input.shape

        # Flatten the image
        return input.flatten()

    def backward(self, gradients, learning_rate):
        # Reshape the image
        return gradients.reshape(self.original_shape)
