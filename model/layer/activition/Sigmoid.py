import numpy
from model.layer.activition.Activation import Activation


class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__()

    def _activate(self, v_x: numpy.ndarray):
        return 1 / (1 + numpy.exp(-v_x))

    def _dActivate(self, v_x: numpy.ndarray):
        return numpy.multiply(self._activate(v_x), 1 - self._activate(v_x))

    def forward(self, input_data: numpy.ndarray):
        self.v_x = input_data
        return self._activate(self.v_x)

    def backward(self, upstream_gradient: numpy.ndarray, learning_rate: float):
        return numpy.multiply(upstream_gradient, self._dActivate(self.v_x))
