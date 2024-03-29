#!/usr/bin/env python3
"""
Module for calculating the cost of a
neural network with L2 regularization
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Function that calculates the cost of a neural
    network with L2 regularization
    cost: a tensor containing the cost of
    the network without L2 regularization
    Returns: a tensor containing the cost of
    the network accounting for L2 regularization
    """
    l2_cost = tf.losses.get_regularization_losses()
    return cost + l2_cost
