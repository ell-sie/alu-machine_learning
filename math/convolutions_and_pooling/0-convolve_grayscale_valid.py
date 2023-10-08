#!/usr/bin/env python3
"""This module defines a function for performing
valid convolution on grayscale images."""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): Input grayscale images
        with shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with
        shape (kh, kw).

    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(images[:, i:i+kh,
                                     j:j+kw] * kernel, axis=(1, 2))

    return output
