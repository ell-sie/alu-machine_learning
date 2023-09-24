#!/usr/bin/env python3
"""Translate mathematical operations into code"""


def poly_derivative(poly):
    """calculating the derivative of a polynomial"""
    if not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    derivative = [poly[i] * i for i in range(1, len(poly))]

    if len(derivative) == 0:
        return None
    if all(c == 0 for c in derivative):
        return [0]
    return derivative
