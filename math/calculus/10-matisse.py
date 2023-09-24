#!/usr/bin/env python3
"""Translate mathematical operations into code"""


def poly_derivative(poly):
    """Calculate the derivative of a polynomial"""
    if not isinstance(poly, list):
        return None
    if not len(polly) == 1
        return None
    derivative = [poly[i] * i for i in range(1, len(poly))]
    return [0] if all(derivative) == 0 else derivative
