#!/usr/bin/env python3
"""a class NeuralNetwork that defines a neural network
    with one hidden layer performing binary classification
"""
import numpy as np


class NeuralNetwork:
    """a class NeuralNetwork that defines a neural network
    with one hidden layer performing binary classification
    """
    def __init__(self, nx, nodes):
        """
        Initializes a NeuralNetwork instance.

        Parameters:
        - nx (int): Number of input features to the neuron.
        - nodes (int): Number of nodes in the hidden layer.

        Raises:
        - TypeError: If nx or nodes is not an integer.
        - ValueError: If nx or nodes is less than 1 (not a positive integer).
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initializing private attributes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter method for the weights vector of the hidden layer.

        Returns:
        - numpy.ndarray: The weights vector of the hidden layer.
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter method for the bias of the hidden layer.

        Returns:
        - numpy.ndarray: The bias of the hidden layer.
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter method for the activated output of the hidden layer.

        Returns:
        - numpy.ndarray: The activated output of the hidden layer.
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter method for the weights vector of the output neuron.

        Returns:
        - numpy.ndarray: The weights vector of the output neuron.
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter method for the bias of the output neuron.

        Returns:
        - float: The bias of the output neuron.
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter method for the activated output of the output neuron.

        Returns:
        - float: The activated output of the output neuron.
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Parameters:
        - X (numpy.ndarray): Input data with shape (nx, m).

        Updates the private attributes __A1 and __A2.
        """
        Z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.A1, self.A2
