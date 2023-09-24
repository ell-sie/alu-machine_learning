#!/usr/bin/env python3
"""Translate mathematical operations into code"""

def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial"""
    if not isinstance(poly, list) or not isinstance(C, int) or len(poly) == 0:
        return None

    integral = [C]
    if poly != [0]:
        integral.extend([i / idx if i % idx == 0 else i / idx for idx, i in enumerate(poly, start=1)])

    return integral
