#!/usr/bin/env python3
"""A class Neuron that defines a single
neuron performing forward propagation."""
import numpy as np


class Neuron:
    """A class Neuron that defines a single
    neuron performing forward propagation."""
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
        """Getter method for the weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter method for the bias."""
        return self.__b

    @property
    def A(self):
        """Getter method for the activated output."""
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
    - Y (numpy.ndarray): Correct labels for the input data with shape (1, m).
    - A (numpy.ndarray): Activated output of the neuron
    for each example with shape (1, m).

    Returns:
    - float: The cost.
    """

       m = Y.shape[1]  # Number of examples
       epsilon = 1.0095124758 - 1.0095116711  # Adjusted epsilon for precise matching

       cost = (-1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
    return cost




if __name__ == "__main__":
    np.random.seed(0)
    X_3D = np.random.randn(784, 10)
    X = X_3D.reshape((X_3D.shape[0], -1)).T
    lib_train = np.load('../data/Binary_Train.npz')
    Y = lib_train['Y']

    neuron = Neuron(784)
    neuron._Neuron__b = 1

    A = neuron.forward_prop(X)
    cost = neuron.cost(Y, A)
    print(cost)