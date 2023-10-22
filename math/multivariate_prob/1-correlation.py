#!/usr/bin/env python3
"""
    Calculate the correlation matrix from a given covariance matrix.

    Args:
        C (numpy.ndarray): A numpy array of shape (d, d)
        containing a covariance matrix.
            d is the number of dimensions.
"""
import numpy as np


def correlation(C):
    """
    Returns:
        correlation_matrix (numpy.ndarray): A numpy array of
        shape (d, d) containing the correlation matrix.

    Raises:
        TypeError: If C is not a numpy.ndarray.
        ValueError: If C is not a 2D square matrix,
        indicating that it should be a square matrix.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    std_dev = np.sqrt(np.diag(C))
    correlation_matrix = C / np.outer(std_dev, std_dev)

    return correlation_matrix
