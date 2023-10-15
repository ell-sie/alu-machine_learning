#!/usr/bin/env python3
"""Module for the functions calc, minor, cofactor
that calculates the cofactor matrix of a matrix:
"""


def calc(matrix):
    """Subdivides a matrix to reach smallest slice required to compute the
    determinant of a bigger matrix.

    Args.
        matrix: A list of lists whose determinant should be calculated.
    Returns:
        The determinant of matrix.
    """

    m_len = len(matrix)
    det = []

    if m_len == 0:
        return 1

    if m_len == 1:
        return matrix[0][0]

    if m_len == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    for i in range(m_len):
        if i == 0:
            det.append(matrix[0][i]
                       * calc([sub[i + 1:] for sub in matrix[i + 1:]]))
        elif i == m_len - 1:
            if i % 2 != 0:
                det.append(-matrix[0][i]
                           * calc([sub[:i] for sub in matrix[1:]]))
            else:
                det.append(matrix[0][i]
                           * calc([sub[:i] for sub in matrix[1:]]))
        else:
            if i % 2 != 0:
                det.append(-matrix[0][i]
                           * calc([sub[:i]
                                  + sub[i + 1:] for sub in matrix[1:]]))
            else:
                det.append(matrix[0][i]
                           * calc([sub[:i]
                                  + sub[i + 1:] for sub in matrix[1:]]))
    return sum(det)


def minor(matrix):
    """Calculates the minor matrix of a matrix

    Args.
        matrix: A list of lists whose minor matrix should be calculated

    Returns:
        The minor matrix of matrix
    """

    m_len = len(matrix)
    if not isinstance(matrix, list) or m_len == 0:
        raise TypeError("matrix must be a list of lists")
    else:
        if not all(isinstance(x, list) for x in matrix):
            raise TypeError("matrix must be a list of lists")
    if not all(m_len == len(x) for x in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    min_matrix = [[] for x in range(m_len)]
    for i in range(m_len):
        for j in range(m_len):
            if i == 0:
                if j == 0:
                    min_matrix[i].append(
                        calc([sub[j + 1:] for sub in matrix[i + 1:]])
                        )
                elif j == m_len - 1:
                    min_matrix[i].append(
                        calc([sub[:j] for sub in matrix[i + 1:]])
                        )
                else:
                    min_matrix[i].append(
                        calc([sub[:j] + sub[j + 1:] for sub in matrix[i + 1:]])
                        )
            elif i == m_len - 1:
                if j == 0:
                    min_matrix[i].append(
                        calc([sub[j + 1:] for sub in matrix[:i]])
                        )
                elif j == m_len - 1:
                    min_matrix[i].append(calc([sub[:j] for sub in matrix[:i]]))
                else:
                    min_matrix[i].append(
                        calc([sub[:j] + sub[j + 1:] for sub in matrix[:i]])
                        )
            else:
                if j == 0:
                    min_matrix[i].append(
                        calc([sub[j + 1:]
                              for sub in matrix[:i] + matrix[i + 1:]])
                        )
                elif j == m_len - 1:
                    min_matrix[i].append(
                        calc([sub[:j] for sub in matrix[:i] + matrix[i + 1:]])
                        )
                else:
                    min_matrix[i].append(
                        calc(
                            [sub[:j] + sub[j + 1:]
                             for sub in matrix[:i] + matrix[i + 1:]]
                            )
                        )

    return min_matrix


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix

    Args.
        matrix: A list of lists whose cofactor matrix should be calculated

    Returns:
        The cofactor matrix of matrix
    """

    min_matrix = minor(matrix)
    cof_matrix = [[] for x in range(len(min_matrix))]
    for i in range(len(min_matrix)):
        if i % 2 == 0:
            cof_matrix[i] = [(x if i % 2 == 0 else -1 * x)
                             for i, x in enumerate(min_matrix[i])]
        else:
            cof_matrix[i] = [(-x if i % 2 == 0 else x)
                             for i, x in enumerate(min_matrix[i])]
    return cof_matrix
