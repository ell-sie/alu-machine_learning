#!/usr/bin/env python3
"""
Module for creating the forward propagation graph
"""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function that returns the prediction of the network in tensor form
    x is the placeholder for the input data
    layer_sizes is a list containing the number of nodes in each layer of the network
    activations is a list containing the activation functions for each layer of the network
    """
    create_layer = __import__('1-create_layer').create_layer
    prediction = x
    for i in range(len(layer_sizes)):
        prediction = create_layer(prediction, layer_sizes[i], activations[i])
    return prediction
