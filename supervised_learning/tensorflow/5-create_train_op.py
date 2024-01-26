#!/usr/bin/env python3
"""
Module for creating the training operation for the network
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Function that returns an operation that trains
    the network using gradient descent
    loss is the loss of the network’s prediction
    alpha is the learning rate
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
