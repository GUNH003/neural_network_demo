import numpy


class OneHot:
    def __init__(self) -> None:
        pass

    def oneHotEncode(self, labels: numpy.ndarray):
        m_one_hot = numpy.zeros((labels.size, numpy.max(labels) + 1))
        for i in range(labels.size):
            m_one_hot[i][labels[i]] = 1.0
        return m_one_hot
