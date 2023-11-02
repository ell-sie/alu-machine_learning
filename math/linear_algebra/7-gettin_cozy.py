#!/usr/bin/env python3
"""
Write a function def cat_matrices2D(mat1, mat2, axis=0):
that concatenates two matrices along a specific axis:
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
You can assume that mat1 and mat2 are 2D matrices containing ints/floats
You can assume all elements in the same dimension are of the same type/shape
You must return a new matrix
If the two matrices cannot be concatenated, return None
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
