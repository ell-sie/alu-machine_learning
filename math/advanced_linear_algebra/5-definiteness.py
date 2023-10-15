#!/usr/bin/env python3
"""Module for the function definiteness
that calculates the definiteness of a matrix
"""

import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix:

    Args.
        matrix: A numpy.ndarray of shape (n, n) whose definiteness should be
        calculated
    Returns.
        The string Positive definite, Positive semi-definite, Negative
        semi-definite, Negative definite, or Indefinite if the matrix is
        positive definite, positive semi-definite, negative semi-definite,
        negative definite of indefinite, respectively.
        If matrix does not fit any of the above categories, return None
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.shape[0] == 0:
        return None

    if any(not isinstance(row, np.ndarray)
           or len(row) != len(matrix) for row in matrix):
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    odds = []
    evens = []
    m_det = 0
    for i in range(matrix.shape[0]):
        if i == 0:
            odds.append(matrix[i][i])
        else:
            if (i + 1) % 2 == 0:
                evens.append(np.linalg.det(matrix[:i + 1][:i + 1]))
                m_det = evens[len(evens) - 1]
            else:
                odds.append(np.linalg.det(matrix[:i + 1][:i + 1]))
                m_det = odds[len(odds) - 1]

    if all(x >= 0 for x in odds + evens):
        if any(x == 0 for x in odds + evens):
            return "Positive semi-definite"
        else:
            return "Positive definite"
    elif all(x <= 0 for x in odds) and all(x >= 0 for x in evens):
        if any(x == 0 for x in odds + evens):
            return "Negative semi-definite"
        else:
            return "Negative definite"
    else:
        return "Indefinite"
