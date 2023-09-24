#!/usr/bin/env python3
"""Translate mathematical operations into code"""

def summation_i_squared(n):
    # Check if n is a valid integer and
    # greater than or equal to 1
    if not isinstance(n, int) or n < 1:
        return None

    # Initialize the sum to 0
    total = 0

    # Iterate from 1 to n, adding the square of each number
    # to the total
    for i in range(1, n + 1):
        total += i**2

    return total
