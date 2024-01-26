#!/usr/bin/env python3
"""
Module for calculating the accuracy of a prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Function that returns the decimal accuracy of the prediction
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
