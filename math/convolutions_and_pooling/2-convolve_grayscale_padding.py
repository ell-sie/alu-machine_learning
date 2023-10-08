#!/usr/bin/env python3
"""This module defines a function for performing
convolution on grayscale images with custom padding."""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Perform convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): Input grayscale images with shape
        (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape
        (kh, kw).
        padding (tuple): Tuple of (ph, pw) where ph is the padding
        for the height and pw is the padding for the width.

    Returns:
        numpy.ndarray: Convolved images with shape (m, h', w'), where
        h' and w' are the height and width of the padded images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad the images with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Calculate output dimensions
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # Initialize the output array
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(padded_images[:, i:i+kh,
                                     j:j+kw] * kernel, axis=(1, 2))

    return output
