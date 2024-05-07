import numpy


class Layer:
    def __init__(self) -> None:
        self.v_x = None

    def forward(self, input_data: numpy.ndarray):
        pass

    def backward(self, upstream_gradient: numpy.ndarray, learning_rate: float):
        pass
