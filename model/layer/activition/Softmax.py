import numpy
from model.layer.activition.Activation import Activation


class Softmax(Activation):
    def __init__(self) -> None:
        super().__init__()

    def _activate(self, v_x: numpy.ndarray):
        shifted = numpy.exp(v_x - numpy.max(v_x, axis=1, keepdims=True))
        return shifted / numpy.sum(shifted, axis=0, keepdims=True)

    def _dActivate(self, v_x: numpy.ndarray):
        v_out = self._activate(v_x)
        n = numpy.size(v_out)
        v_temp = numpy.tile(v_out, n)
        return numpy.multiply(v_temp, (numpy.identity(n) - v_temp.T))

    def forward(self, input_data: numpy.ndarray):
        self.v_x = input_data
        return self._activate(self.v_x)

    def backward(self, upstream_gradient: numpy.ndarray, learning_rate: float):
        return numpy.dot(self._dActivate(self.v_x), upstream_gradient)
