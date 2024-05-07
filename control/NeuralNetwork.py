import numpy


class NeuralNetwork:
    def __init__(self, layers: list) -> None:
        self.layers = layers

    def forward(self, trainning_data: numpy.ndarray):
        running_output = trainning_data
        for layer in self.layers:
            temp = layer.forward(running_output)
            running_output = temp
        return running_output

    def backward(self, gradient: numpy.ndarray, learning_rate: float):
        running_gradient = gradient
        for i in range(len(self.layers)-1, -1, -1):
            temp = self.layers[i].backward(running_gradient, learning_rate)
            running_gradient = temp
        return running_gradient
