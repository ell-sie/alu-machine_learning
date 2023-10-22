#!/usr/bin/env python3
"""
    Calculate the mean and covariance of a data set.

    Args:
        X (numpy.ndarray): A 2D numpy array of shape
        (n, d) containing the data set.
            n is the number of data points, and d is
            the number of dimensions in each data point.
"""
import numpy as np


def mean_cov(X):
    """
    Returns:
        mean (numpy.ndarray): A 2D numpy array of shape (1, d)
        containing the mean of the data set.
        cov (numpy.ndarray): A 2D numpy array of shape (d, d)
        containing the covariance matrix of the data set.

    Raises:
        TypeError: If X is not a 2D numpy array.
        ValueError: If n is less than 2, indicating
        that X must contain multiple data points.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.dot((X - mean).T, X - mean) / (X.shape[0] - 1)

    return mean, cov
