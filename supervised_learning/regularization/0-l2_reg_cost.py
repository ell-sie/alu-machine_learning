#!/usr/bin/env python3
"""
Module for calculating the cost of a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Function that calculates the cost of a
    neural network with L2 regularization
    cost: the cost of the network without L2 regularization
    lambtha: the regularization parameter
    weights: a dictionary of the weights
    and biases (numpy.ndarrays) of the neural network
    L: the number of layers in the neural network
    m: the number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """
    L2_cost = 0
    for i in range(1, L + 1):
        L2_cost += np.linalg.norm(weights['W' + str(i)])
    L2_cost *= lambtha / (2 * m)
    return cost + L2_cost
