#!/usr/bin/env python3
def cofactor(matrix):
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            cofactor_row.append(determinant(sub_matrix) * (-1) ** (i + j))
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
