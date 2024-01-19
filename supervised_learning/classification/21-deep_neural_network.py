#!/usr/bin/env python3
"""a class DeepNeuralNetwork that defines a deep neural network
    performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """a class DeepNeuralNetwork that defines a deep neural network
    performing binary classification
    """
    def __init__(self, nx, layers):
        """
        Initializes a DeepNeuralNetwork instance.

        Parameters:
        - nx (int): Number of input features to the neuron.
        - layers (list): List representing the number
        of nodes in each layer of the network.

        Raises:
        - TypeError: If nx is not an integer or layers is not a list.
        - ValueError: If nx is less than 1 or layers is not a positive list.

        Sets the private instance attributes:
        - __L: The number of layers in the neural network.
        - __cache: A dictionary to hold all intermediary values of the network.
        - __weights: A dictionary to hold all weights
        and biased of the network.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                self.__weights['W' + str(i + 1)] = np.random.randn(layers[i],
                                                    nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i + 1)] = np.random.randn(layers[i],
                                                    layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        Getter method for the number of layers.

        Returns:
        - int: The number of layers in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter method for all intermediary values of the network.

        Returns:
        - dict: A dictionary to hold all intermediary values of the network.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter method for all weights and biased of the network.

        Returns:
        - dict: A dictionary to hold all weights and biased of the network.
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Parameters:
        - X (numpy.ndarray): Input data with shape (nx, m).

        Updates the private attribute __cache.
        Returns the output of the neural network and the cache, respectively.
        """
        self.__cache['A0'] = X
        for i in range(self.__L):
            Z = np.dot(self.__weights['W' + str(i + 1)],
            self.__cache['A' + str(i)]) + self.__weights['b' + str(i + 1)]
            self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(-Z))
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
        - Y (numpy.ndarray): Correct labels for the input data.
        - A (numpy.ndarray): Activated output of the neuron for each example.

        Returns:
        - float: The cost of the model.
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Parameters:
        - X (numpy.ndarray): Input data with shape (nx, m).
        - Y (numpy.ndarray): Correct labels for the input data.

        Returns:
        - numpy.ndarray: The neuron’s prediction.
        - float: The cost of the network.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Parameters:
        - Y (numpy.ndarray): Correct labels for the input data.
        - cache (dict): Intermediary values of the network.
        - alpha (float): Learning rate.

        Updates the private attribute __weights.
        """
        m = Y.shape[1]
        dz = cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            dw = np.matmul(cache['A' + str(i - 1)], dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            dz = np.matmul(self.__weights['W' + str(i)].T, dz) * \
                (cache['A' + str(i - 1)] * (1 - cache['A' + str(i - 1)]))
            self.__weights['W' + str(i)] -= alpha * dw.T
            self.__weights['b' + str(i)] -= alpha * db
