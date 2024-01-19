#!/usr/bin/env python3
"""a class Neuron that defines a single neuron
    perfoming forward propagation
"""
import numpy as np


class Neuron:
    """a class Neuron that defines a single neuron
    perfoming forward propagation
    """
    def __init__(self, nx):
        """
        Initializes a Neuron instance.

        Parameters:
        - nx (int): Number of input features to the neuron.

        Raises:
        - TypeError: If nx is not an integer.
        - ValueError: If nx is less than 1 (not a positive integer).
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initializing private attributes
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter method for the weights vector.

        Returns:
        - numpy.ndarray: The weights vector.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter method for the bias.

        Returns:
        - float: The bias value.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter method for the activated output.

        Returns:
        - float: The activated output.
        """
        return self.__A

    def forward_prop(self, X):
        """
        Performs forward propagation for the neuron.

        Parameters:
        - X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
        - numpy.ndarray: The activated output (__A).
        """
        Z = np.dot(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
        - Y (numpy.ndarray): Correct labels for the input data.
        - A (numpy.ndarray): Activated output of the neuron for each example.

        Returns:
        - float: The cost.
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Parameters:
        - X (numpy.ndarray): Input data.
        - Y (numpy.ndarray): Correct labels for the input data.

        Returns:
        - numpy.ndarray: The neuron's prediction.
        - float: The cost of the network.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Parameters:
        - X (numpy.ndarray): Input data.
        - Y (numpy.ndarray): Correct labels for the input data.
        - A (numpy.ndarray): Activated output of the neuron for each example.
        - alpha (float): Learning rate.

        Updates the private attributes __W and __b.
        """
        m = Y.shape[1]
        dz = A - Y
        dw = np.dot(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron.

        Parameters:
        - X (numpy.ndarray): Input data.
        - Y (numpy.ndarray): Correct labels for the input data.
        - iterations (int): Number of iterations to train over.
        - alpha (float): Learning rate.

        Updates the private attributes __W, __b, and __A.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)


if __name__ == "__main__":
    np.random.seed(0)
    lib_train = np.load('../data/Binary_Train.npz')
    X_train_3D, Y_train = lib_train['X'], lib_train['Y']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    lib_dev = np.load('../data/Binary_Dev.npz')
    X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
    X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T
    neuron = Neuron(X_train.shape[0])
    neuron._Neuron__b = 1
    A, cost = neuron.train(X_train, Y_train, iterations=10)
    print("Train cost:", np.round(cost, decimals=10))
    print("Train data:", np.round(A, decimals=10))
    print("Train Neuron A:", np.round(neuron.A, decimals=10))
    A, cost = neuron.evaluate(X_dev, Y_dev)
    print("Dev cost:", np.round(cost, decimals=10))
    print("Dev data:", np.round(A, decimals=10))
    print("Dev Neuron A:", np.round(neuron.A, decimals=10))
