#!/usr/bin/env python3
"""
Module for creating placeholders
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Function that returns two placeholders, x and y, for the neural network
    nx: the number of feature columns in our data
    classes: the number of classes in our classifier
    Returns: placeholders named x and y, respectively
    """
    x = tf.placeholder(dtype=tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, classes), name='y')
    return x, y
