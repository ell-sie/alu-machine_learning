#!/usr/bin/env python3
"""
Module for conducting forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Function that conducts forward propagation using Dropout
    X: a numpy.ndarray of shape (nx, m) containing the input data for the network
    weights: a dictionary of the weights and biases of the neural network
    L: the number of layers in the network
    keep_prob: the probability that a node will be kept
    Returns: a dictionary containing the outputs
    of each layer and the dropout mask used on each layer
    """
    cache = {}
    cache["A0"] = X
    for i in range(1, L + 1):
        Z = np.dot(weights["W{}".format(i)],
                   cache["A{}".format(
                    i - 1)]) + weights["b{}".format(i)]
        if i != L:
            tanh = np.sinh(Z) / np.cosh(Z)
            D1 = np.random.rand(tanh.shape[0], tanh.shape[1])
            D1 = (D1 < keep_prob).astype(int)
            tanh = tanh * D1
            tanh = tanh / keep_prob
            cache["D{}".format(i)] = D1
        else:
            t = np.exp(Z)
            tanh = np.exp(Z) / np.sum(t, axis=0, keepdims=True)
        cache["A{}".format(i)] = tanh
    return cache
