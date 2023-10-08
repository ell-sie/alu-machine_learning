#!/usr/bin/env python3
"""This module defines a function for performing pooling
on images with various options."""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Perform pooling on images.

    Args:
        images (numpy.ndarray): Input images with shape
        (m, h, w, c).
        kernel_shape (tuple): Pooling kernel shape
        (kh, kw).
        stride (tuple): Stride options: (sh, sw).
        mode (str): Pooling mode ('max' or 'avg').

    Returns:
        numpy.ndarray: Pooled images with shape
        (m, h', w', c), where h' and w' are the height
        and width of the output images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Initialize the output array
    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    images[:, i * sh:i * sh + kh,
                           j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(images[:, i * sh:i * sh + kh,
                                                    j * sw:j * sw + kw, :],
                                             axis=(1, 2))

    return output
