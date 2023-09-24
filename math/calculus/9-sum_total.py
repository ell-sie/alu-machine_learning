#!/usr/bin/env python3
"""Translate mathematical summation into code"""

def calculate_sum_of_squares(n):
    """Calculate the sum of squares from 1 to n"""
    if not isinstance(n, int) or n < 1:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
