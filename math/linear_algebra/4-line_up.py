#!/usr/bin/env python3
"""
Writing a that function def add_arrays(arr1, arr2):
that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
You can assume that arr1 and arr2 are lists of ints/floats
You must return a new list
If arr1 and arr2 are not the same shape, return None
    """
    if len(arr1) != len(arr2):
        return None
    return [x + y for x, y in zip(arr1, arr2)]
