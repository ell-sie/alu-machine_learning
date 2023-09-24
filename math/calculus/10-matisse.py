#!/usr/bin/env python3
"""Translate mathematical operations into code"""

def poly_derivative(poly):
    """Calculate the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 1:
        return [0]

    derivative = [poly[i] * i for i in range(1, len(poly))]

    return derivative if any(derivative) else [0]
