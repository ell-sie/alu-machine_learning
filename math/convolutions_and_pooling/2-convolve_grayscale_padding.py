#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    """
    Perform a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): Grayscale images of shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel of shape (kh, kw).
        padding (tuple): Padding as (ph, pw).

    Returns:
        numpy.ndarray: Convolved images with shape (m, h + 2*ph, w + 2*pw).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    output = np.zeros((m, output_h, output_w), dtype=np.int32)

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(images_padded[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output
