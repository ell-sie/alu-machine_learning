#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Perform a same convolution on grayscale images.

    Args:
        images (numpy.ndarray): Grayscale images of shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel of shape (kh, kw).

    Returns:
        numpy.ndarray: Convolved images with shape (m, h, w).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    images_padded = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros((m, h, w), dtype=np.int32)

    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(images_padded[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output
