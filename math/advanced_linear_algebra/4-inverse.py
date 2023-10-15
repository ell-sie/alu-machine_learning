#!/usr/bin/env python3
import numpy as np

def inverse(matrix):
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adjugate_matrix = adjugate(matrix)
    inverse_matrix = [[adjugate_matrix[i][j] / det for j in range(n)] for i in range(n)]

    return inverse_matrix
