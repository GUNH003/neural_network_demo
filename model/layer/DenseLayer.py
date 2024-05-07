import numpy
from model.layer.Layer import Layer

RAND_COEFFICIENT = 0.1


class DenseLayer(Layer):
    def __init__(self, num_neurons: int, num_traits: int) -> None:
        super().__init__()
        self.m_W = self._makeRandMatrix(num_neurons, num_traits)
        self.v_b = self._makeRandMatrix(num_neurons, 1)

    def _makeRandMatrix(self, num_rows: int, num_cols: int) -> numpy.ndarray:
        return RAND_COEFFICIENT * numpy.random.randn(num_rows, num_cols)

    def _update(self, parameter: numpy.ndarray, gradient: numpy.ndarray,
                learning_rate: float) -> None:
        parameter -= learning_rate * gradient

    def forward(self, input_data: numpy.ndarray):
        self.v_x = input_data
        return numpy.dot(self.m_W, self.v_x) + self.v_b

    def backward(self, upstream_gradient: numpy.ndarray, learning_rate: float):
        m_local_gradient = numpy.dot(upstream_gradient, self.v_x.T)
        self._update(self.m_W, m_local_gradient, learning_rate)
        self._update(self.v_b, upstream_gradient, learning_rate)
        return numpy.dot(self.m_W.T, upstream_gradient)
