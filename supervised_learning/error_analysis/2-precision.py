#!/usr/bin/env python3
"""
Calculates the precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    a function that calculates the precision for each
    class in a confusion matrix
    :param confusion: a confusion numpy.ndarray of shape
    (classes, classes) where row indices represent the correct
    labels and column indices represent the predicted labels
    :return: a numpy.ndarray of shape (classes,)
    containing the precision of each class
    """
    true_positives = np.diagonal(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    precision = true_positives / predicted_positives

    return precision
