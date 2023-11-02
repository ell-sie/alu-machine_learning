#!/usr/bin/env python3
import numpy as np
"""
Write a function def np_cat(mat1, mat2, axis=0)
that concatenates two matrices along a specific axis:
"""


def np_cat(mat1, mat2, axis=0):
    """
    You can assume that mat1 and mat2 can be interpreted as numpy.ndarrays
You must return a new numpy.ndarray
You are not allowed to use any loops or conditional statements
You may use: import numpy as np
You can assume that mat1 and mat2 are never empty
    """
    return np.concatenate((mat1, mat2), axis=axis)
