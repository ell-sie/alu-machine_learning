#!/usr/bin/env python3
"""
a class Neuron that defines a single neuron
performing binary classification

"""
import numpy as np

class Neuron:
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

        # Initializing weights, bias, and activated output
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

    @A.setter
    def A(self, value):
        """
        Setter method for the activated output.

        Parameters:
        - value (float): The new value for the activated output.
        """
        self.__A = value

if __name__ == "__main__":
    np.random.seed(0)
    neuron = Neuron(784)
    print(neuron.W)
    print(neuron.W.shape)
    print(neuron.b)
    print(neuron.A)
    neuron.A = 10
    print(neuron.A)
