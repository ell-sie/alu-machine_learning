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
    cache['A0'] = X
    for i in range(1, L + 1):
        Z = np.matmul(weights['W' + str(i)],\
            cache['A' + str(i - 1)]) + weights['b' + str(i)]
        if i != L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = np.multiply(A, D)
            A /= keep_prob
            cache['D' + str(i)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        cache['A' + str(i)] = A
    return cache
