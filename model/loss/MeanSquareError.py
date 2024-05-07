import numpy
from model.loss.Loss import Loss


class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predicted, actual) -> float:
        return numpy.mean(numpy.sum((predicted - actual) ** 2, axis=0), axis=0)

    def backward(self, predicted, actual) -> numpy.ndarray:
        return 2 * (predicted - actual) / actual.shape[1]
