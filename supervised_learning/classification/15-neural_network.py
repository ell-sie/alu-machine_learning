#!/usr/bin/env python3
"""a class NeuralNetwork that defines a neural network
    with one hidden layer performing binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


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
        Evaluates the neural network’s predictions.

        Parameters:
        - X (numpy.ndarray): Input data with shape (nx, m).
        - Y (numpy.ndarray): Correct labels for the input data.

        Returns:
        - numpy.ndarray: The neuron’s prediction.
        - float: The cost of the network.
        """
        self.forward_prop(X)
        prediction = np.where(self.A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Parameters:
        - X (numpy.ndarray): Input data with shape (nx, m).
        - Y (numpy.ndarray): Correct labels for the input data.
        - A1 (numpy.ndarray): Output of the hidden layer.
        - A2 (numpy.ndarray): Predicted output.
        - alpha (float): Learning rate.

        Updates the private attributes __W1, __b1, __W2, and __b2.
        """
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neural network.

        Parameters:
        - X (numpy.ndarray): Input data with shape (nx, m).
        - Y (numpy.ndarray): Correct labels for the input data.
        - iterations (int): Number of iterations to train over.
        - alpha (float): Learning rate.
        - verbose (bool): Defines whether or not to print information about the training.
        - graph (bool): Defines whether or not to graph information about the training.
        - step (int): Interval for printing data to stdout or plotting.

        Raises:
        - TypeError: If iterations is not an integer or alpha is not a float or step is not an integer.
        - ValueError: If iterations or alpha or step is not positive.

        Updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2.
        Returns the evaluation of the training data after iterations of training have occurred.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        if verbose or graph:
            costs = []
            steps = []

        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.A2)
            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))
            if graph and i % step == 0:
                costs.append(cost)
                steps.append(i)
            if i < iterations:
                self.gradient_descent(X, Y, self.A1, self.A2, alpha)

        if graph:
            plt.plot(steps, costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
