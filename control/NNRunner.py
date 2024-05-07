import numpy
from model.loss.Loss import Loss
from control.NeuralNetwork import NeuralNetwork


class NNRunner:
    def __init__(self,
                 x_train: numpy.ndarray,
                 y_train: numpy.ndarray,
                 layers: list,
                 loss_function: Loss) -> None:
        self.m_X = x_train
        self.m_Y = y_train
        self.NN = NeuralNetwork(layers)
        self.loss = loss_function

    def _isPredCorrect(self,
                       predicted: numpy.ndarray,
                       actual: numpy.ndarray) -> int:
        return numpy.argmax(predicted) == numpy.argmax(actual)

    def train(self, epochs=3, learning_rate=1e-2):
        print("Training in progress...")
        for i in range(epochs):
            print(f"Epoch: {i + 1}")
            s_error = 0.0
            s_correct_pred = 0
            for v_x, v_y in zip(self.m_X, self.m_Y):
                v_x = v_x.reshape(-1, 1)
                v_y = v_y.reshape(-1, 1)
                v_prediction = self.NN.forward(v_x)
                s_error += self.loss.forward(v_prediction, v_y)
                s_correct_pred += self._isPredCorrect(v_prediction, v_y)
                v_gradient = self.loss.backward(v_prediction, v_y)
                self.NN.backward(v_gradient, learning_rate)
            s_e = s_error / self.m_X.shape[0]
            s_acc = s_correct_pred / self.m_X.shape[0]
            print(f"Error: {s_e}, Accuracy: {s_acc}")
        print("Training completed.")

    def test(self, x_test: numpy.ndarray, y_test: numpy.ndarray):
        print("Testing in progress...")
        s_error = 0.0
        s_correct_pred = 0
        for v_x, v_y in zip(x_test, y_test):
            v_x = v_x.reshape(-1, 1)
            v_y = v_y.reshape(-1, 1)
            v_prediction = self.NN.forward(v_x)
            s_error += self.loss.forward(v_prediction, v_y)
            s_correct_pred += self._isPredCorrect(v_prediction, v_y)
        s_e = s_error / x_test.shape[0]
        s_acc = s_correct_pred / x_test.shape[0]
        print(f"TestError: {s_e}, TestAccuracy: {s_acc}")
        print("Testing completed.")
