#!/usr/bin/env python3
import numpy as np

def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Perform a convolution on images with channels, with optional padding and stride.

    Args:
        images (numpy.ndarray): Images of shape (m, h, w, c).
        kernel (numpy.ndarray): Convolution kernel of shape (kh, kw, c).
        padding (str or tuple): Padding, 'same', 'valid', or (ph, pw).
        stride (tuple): Stride as (sh, sw).

    Returns:
        numpy.ndarray: Convolved images with shape (m, output_h, output_w, nc).
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernel.shape

    if padding == 'same':
        ph = max((h - 1) * stride[0] + kh - h, 0)
        pw = max((w - 1) * stride[1] + kw - w, 0)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    sh, sw = stride
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, nc), dtype=np.int32)

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j, :] = np.sum(images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] * kernel, axis=(1, 2, 3))

    return output
