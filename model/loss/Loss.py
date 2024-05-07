import numpy


class Loss:
    def __init__(self) -> None:
        pass

    def forward(self, predicted, actual) -> float:
        pass

    def backward(self, predicted, actual) -> numpy.ndarray:
        pass
