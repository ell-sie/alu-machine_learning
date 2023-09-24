#!/usr/bin/env python3
"""Translate mathematical operations into code"""


def poly_derivative(poly):
    """Calculate the derivative of a polynomial"""
    if not isinstance(poly, list):
        return None
    derivative = [poly[i] * i for i in range(1, len(poly))]
    return [0] if all(c == 0 for c in derivative) else derivative
