#!/usr/bin/env python3
import numpy as np

def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None
    return np.dot(mat1, mat2)
