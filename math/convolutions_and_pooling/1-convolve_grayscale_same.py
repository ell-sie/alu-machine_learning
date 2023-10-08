#!/usr/bin/env python3
"""This module defines a function for performing
a same convolution on grayscale images"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Perform a same convolution on grayscale images with padding.

    Args:
        images (numpy.ndarray): Input grayscale images with shape
        (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape
        (kh, kw).

    Returns:
        numpy.ndarray: Convolved images with shape (m, h, w), where
        h and w are the height and width of the input images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad the images with zeros
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h),
                                    (pad_w, pad_w)), mode='constant')

    # Initialize the output array
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(padded_images[:, i:i+kh,
                                     j:j+kw] * kernel, axis=(1, 2))

    return output
