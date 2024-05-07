import numpy


class TransformData:
    def __init__(self) -> None:
        pass

    def standardize(self, input: numpy.ndarray):
        m = numpy.mean(input.astype("float32"))
        sd = numpy.std(input.astype("float32"))
        return (input - m) / sd

    def compress(self, input: numpy.ndarray, num_traits: int):
        return input.astype("float32") / num_traits

    def flatten(self, input: numpy.ndarray):
        return numpy.reshape(input,
                             (input.shape[0], input.shape[1] * input.shape[2]))
