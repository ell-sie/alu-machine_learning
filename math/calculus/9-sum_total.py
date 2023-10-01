#!/usr/bin/env python3
"""Translate mathematical operations into code"""


def summation_i_squared(n):
    """calculates sum_{i=1}^{n} i^2"""
    if type(n) != int or n < 1:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
