#!/usr/bin/env python3
"""Translate mathematical operations into code"""

def summation_i_squared(n):
    # Check if n is a valid integer and greater 
    # than or equal to 1
    if not isinstance(n, int) or n < 1:
        return None

    # Use the formula for the sum of squares
    result = (n * (n + 1) * (2 * n + 1)) // 6

    return result
