#!/usr/bin/env python3
"""
Write a function def mat_mul(mat1, mat2): that performs matrix multiplication
"""


def matrix_multiply(matrix1, matrix2):
    """
You can assume that mat1 and mat2 are 2D matrices containing ints/floats
You can assume all elements in the same dimension are of the same type/shape
You must return a new matrix
If the two matrices cannot be multiplied, return None
    """
    if len(matrix1[0]) != len(matrix2):
        return None
    # Determine the dimensions of the resulting matrix
    num_rows = len(matrix1)
    num_columns = len(matrix2[0])
    result_matrix = [[0] * num_columns for _ in range(num_rows)]

    # Perform matrix multiplication
    for i in range(num_rows):
        for j in range(num_columns):
            for k in range(len(matrix2)):
                result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]

    return result_matrix
