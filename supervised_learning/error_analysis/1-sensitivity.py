#!/usr/bin/env python3
"""
Calculates the sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    a function that calculates the sensitivity
    for each class in a confusion matrix
    :param confusion: a confusion numpy.ndarray of
    shape (classes, classes) where row indices represent the
    correct labels and column indices represent the predicted labels
    :return: a numpy.ndarray of shape (classes,)
    containing the sensitivity of each class
    """
    true_positives = np.diagonal(confusion)
    actual_positives = np.sum(confusion, axis=1)
    sensitivity = true_positives / actual_positives

    return sensitivity
