#!/usr/bin/env python3
"""
Creates a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    a function that creates a confusion matrix
    :param labels: one-hot numpy.ndarray of shape (m, classes)
    containing the correct labels for each data point
    :param logits: one-hot numpy.ndarray of shape (m, classes)
    containing the predicted labels
    :return: a confusion numpy.ndarray of shape (classes, classes)
    with row indices representing the correct labels and column
    indices representing the predicted labels
    """
    classes = labels.shape[1]
    confusion_matrix = np.zeros((classes, classes))

    actual = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)

    for a, p in zip(actual, predicted):
        confusion_matrix[a][p] += 1

    return confusion_matrix
