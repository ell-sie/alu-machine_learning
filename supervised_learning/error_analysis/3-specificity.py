#!/usr/bin/env python3
"""
Calculates the specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    a function that calculates the specificity
    for each class in a confusion matrix
    :param confusion: a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column
    indices represent the predicted labels
    :return: a numpy.ndarray of shape (classes,)
    containing the specificity of each class
    """
    true_negatives = np.sum(confusion) - np.sum(confusion, axis=0) -\
                    np.sum(confusion, axis=1) + np.diagonal(confusion)
    actual_negatives = np.sum(confusion) - np.sum(confusion, axis=1)
    specificity = true_negatives / actual_negatives

    return specificity
