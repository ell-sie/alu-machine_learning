#!/usr/bin/env python3
import numpy as np

def definiteness(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if not matrix.shape:
        return None

    eig_values = np.linalg.eigvals(matrix)
    pos_count = np.sum(eig_values > 0)
    neg_count = np.sum(eig_values < 0)
    zero_count = np.sum(eig_values == 0)

    if zero_count == matrix.shape[0]:
        return "Indefinite"
    elif pos_count == matrix.shape[0]:
        return "Positive definite"
    elif pos_count == matrix.shape[0] - zero_count:
        return "Positive semi-definite"
    elif neg_count == matrix.shape[0]:
        return "Negative definite"
    elif neg_count == matrix.shape[0] - zero_count:
        return "Negative semi-definite"
    else:
        return None
