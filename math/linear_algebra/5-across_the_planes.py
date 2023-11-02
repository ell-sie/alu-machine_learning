#!/usr/bin/env python3
"""
function def add_matrices2D(mat1, mat2):
that adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
You can assume that mat1 and mat2 are 2D matrices containing ints/floats
You can assume all elements in the same dimension are of the same type/shape
You must return a new matrix
If mat1 and mat2 are not the same shape, return None
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [
        [
            mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))
        ] for i in range(len(mat1))]
