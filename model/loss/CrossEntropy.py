import numpy
from model.loss.Loss import Loss


class CE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predicted, actual) -> float:
        clipped = numpy.clip(predicted, 1e-10, 1-1e-10)
        v_error = -1 * numpy.sum(numpy.multiply(actual, numpy.log(clipped)),
                                 axis=1,
                                 keepdims=True)
        return numpy.mean(v_error)

    def backward(self, predicted, actual) -> numpy.ndarray:
        return predicted - actual
