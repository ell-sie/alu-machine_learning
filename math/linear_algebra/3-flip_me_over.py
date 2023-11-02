#!/usr/bin/env python3
def matrix_transpose(matrix):
    # Check if the input matrix is empty
    if not matrix:
        return []

    # Determine the number of rows and columns in the input matrix
    num_rows = len(matrix)
    num_columns = len(matrix[0])

    # Create a new matrix to store the transpose
    transpose = [[0] * num_rows for _ in range(num_columns)]

    # Iterate through the input matrix and fill in the transpose
    for i in range(num_rows):
        for j in range(num_columns):
            transpose[j][i] = matrix[i][j]

    return transpose
