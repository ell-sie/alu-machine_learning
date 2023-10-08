#!/usr/bin/env python3
import numpy as np

def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Perform a convolution on images using multiple kernels, with optional padding and stride.

    Args:
        images (numpy.ndarray): Images of shape (m, h, w, c).
        kernels (numpy.ndarray): Convolution kernels of shape (kh, kw, c, nc).
        padding (str or tuple): Padding, 'same', 'valid', or (ph, pw).
        stride (tuple): Stride as (sh, sw).

    Returns:
        numpy.ndarray: Convolved images with shape (m, output_h, output_w, nc).
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels
