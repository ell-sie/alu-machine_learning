#!/usr/bin/env python3
"""Translate mathematical operations into code"""


"""Translate mathematical operations into code"""

def poly_derivative(poly):
    """Calculate the derivative of a polynomial"""
    if not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    derivative = [poly[i] * i for i in range(1, len(poly))]
    if len(derivative) == 0:
        return [0]
    return [0] if all(derivative) == 0 else derivative
