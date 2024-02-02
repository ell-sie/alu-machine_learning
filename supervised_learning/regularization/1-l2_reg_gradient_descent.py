#!/usr/bin/env python3
"""
Module for updating the weights and biases of a
neural network using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function that updates the weights and biases of a neural
    network using gradient descent with L2 regularization
    Y: a one-hot numpy.ndarray of shape (classes, m) that
    contains the correct labels for the data
    weights: a dictionary of the weights and biases of the neural network
    cache: a dictionary of the outputs of each layer of the neural network
    alpha: the learning rate
    lambtha: the L2 regularization parameter
    L: the number of layers of the network
    The neural network uses tanh activations on each
    layer except the last, which uses a softmax activation
    The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        dw = (1 / m) * np.matmul(dz, cache['A' + str(i - 1)].T) +\
             ((lambtha / m) * weights['W' + str(i)])
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(weights['W' + str(i)].T, dz) * (1 - cache['A' + str(i - 1)] ** 2)
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
