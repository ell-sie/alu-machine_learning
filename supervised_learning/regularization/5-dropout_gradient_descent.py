#!/usr/bin/env python3
"""
Module for updating the weights of a neural
network with Dropout regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Function that updates the weights of a neural
    network with Dropout regularization using gradient descent
    Y: a one-hot numpy.ndarray of shape (classes, m)
    that contains the correct labels for the data
    weights: a dictionary of the weights and biases of the neural network
    cache: a dictionary of the outputs and dropout
    masks of each layer of the neural network
    alpha: the learning rate
    keep_prob: the probability that a node will be kept
    L: the number of layers of the network
    The weights of the network should be updated in place
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        dw = (1 / m) * np.matmul(dz, cache['A' + str(i - 1)].T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        if i > 1:
            dz = np.matmul(weights['W' + str(i)].T, dz) * (1 - cache['A' + str(i - 1)] ** 2)
            dz *= cache['D' + str(i - 1)]
            dz /= keep_prob
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
