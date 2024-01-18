#!/usr/bin/env python3
"""a class Neuron that defines a single neuron
    perfoming binary classification
"""
import numpy as np


class Neuron:
    """a class Neuron that defines a single neuron
    perfoming binary classification
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
        - Y (numpy.ndarray): Correct labels with shape (1, m).
        - A (numpy.ndarray): Activated output with shape (1, m).

        Returns:
        - float: The cost.
        """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A + 1.0000001) + (1 - Y) * np.log(1.0000001 - A))
        return cost


if __name__ == "__main__":
    # Example usage and testing
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y']
    X = X_3D.reshape((X_3D.shape[0], -1)).T

    np.random.seed(0)
    neuron = Neuron(X.shape[0])
    A = neuron.forward_prop(X)
    cost = neuron.cost(Y, A)
    print(cost)
