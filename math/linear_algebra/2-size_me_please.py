#!/usr/bin/env python3

"""
This module defines a function matrix_shape that calculates
the shape of a matrix.
"""


def matrix_shape(matrix):
    '''
The function takes a matrix as input and returns a list of
integers representing the shape of the matrix. It counts the
dimensions of the matrix, including nested lists.

Example:
mat = [[1, 2, 3], [4, 5, 6]]
shape = matrix_shape(mat)  # Returns [2, 3] as there are
2 rows and 3 columns.
'''

    shape = []

    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else None

    return shape


if __name__ == "__main__":
    matrix_shape_function = __import__('2-size_me_please').matrix_shape
