import keras
from control.NNRunner import NNRunner
from model.layer.DenseLayer import DenseLayer
from model.layer.activition.Sigmoid import Sigmoid
from model.loss.MeanSquareError import MSE
from utils.OneHot import OneHot
from utils.TransformData import TransformData

if __name__ == "__main__":

    """
    Reads data file.
    """
    data = keras.datasets.mnist.load_data()
    # Training data
    x_train = data[0][0]
    y_train = data[0][1]
    # Testing data
    x_test = data[1][0]
    y_test = data[1][1]

    """
    Prepares data.
    """
    # Flattens data and compress training data ([0, 1])
    x_train = TransformData().flatten(x_train)
    x_train = TransformData().compress(x_train, x_train.shape[1])
    # One-hot encodes training labels
    y_train = OneHot().oneHotEncode(y_train)

    # Flattens data and compress testing data ([0, 1])
    x_test = TransformData().flatten(x_test)
    x_test = TransformData().compress(x_test, x_test.shape[1])
    # One-hot encodes testing labels
    y_test = OneHot().oneHotEncode(y_test)

    """
    Creates neural network.
    """
    # Creates layers
    layers = [
        DenseLayer(16, 784),  # input layer with 16 neurons
        Sigmoid(),            # activation for input layer
        DenseLayer(10, 16),   # output layer with 10 neurons
        Sigmoid()             # activation for ouput layer
    ]

    # Create loss function
    loss_function = MSE()     # mean square error as loss

    # Create network runner
    runner = NNRunner(x_train, y_train, layers, loss_function)

    # Trains the model
    runner.train(epochs=3,
                 learning_rate=0.1)

    # Predicts testing data output
    runner.test(x_test, y_test)
