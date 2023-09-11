#!/usr/bin/env python3

def matrix_shape(matrix):
    shape = []

    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else None

    return shape


if __name__ == "__main__":
    matrix_shape_function = __import__('2-size_me_please').matrix_shape
