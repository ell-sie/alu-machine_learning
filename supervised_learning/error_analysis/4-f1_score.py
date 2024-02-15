#!/usr/bin/env python3
"""
Calculates the F1 score of a confusion matrix
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    a function that calculates the F1 score
    for each class in a confusion matrix
    :param confusion: a confusion numpy.ndarray of shape
    (classes, classes) where row indices represent the correct
    labels and column indices represent the predicted labels
    :return: a numpy.ndarray of shape (classes,)
    containing the F1 score of each class
    """
    precision_values = precision(confusion)
    sensitivity_values = sensitivity(confusion)
    f1 = 2 * (precision_values * sensitivity_values) /\
        (precision_values + sensitivity_values)

    return f1
