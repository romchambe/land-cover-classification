class Reshape:
    def __init__(self, input_shape, output_shape) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return input.reshape(self.output_shape)

    def backward(self, gradients, learning_rate):
        return gradients.reshape(self.input_shape)
