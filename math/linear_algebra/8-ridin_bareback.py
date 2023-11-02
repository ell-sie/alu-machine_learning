#!/usr/bin/env python3
import numpy as np
"""
Write a function def mat_mul(mat1, mat2): that performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
You can assume that mat1 and mat2 are 2D matrices containing ints/floats
You can assume all elements in the same dimension are of the same type/shape
You must return a new matrix
If the two matrices cannot be multiplied, return None
    """
    if len(mat1[0]) != len(mat2):
        return None
    return np.dot(mat1, mat2)
