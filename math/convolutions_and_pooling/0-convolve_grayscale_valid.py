#!/usr/bin/env python3
"""
Convolution on grayscale images using a valid convolution.

This script defines a function `convolve_grayscale_valid`
that performs a valid convolution on grayscale images
using a given kernel.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] *
                                    kernel, axis=(1, 2))

    return output
