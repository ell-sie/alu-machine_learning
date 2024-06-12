#!/usr/bin/env python3
"""
A function that calculates the positional encoding for a transformer.
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.

    Parameters:
    - max_seq_len (int): Maximum sequence length.
    - dm (int): Model depth.

    Returns:
    - numpy.ndarray: Positional encoding vectors of shape (max_seq_len, dm).
    """
    # Initialize the positional encoding matrix
    PE = np.zeros((max_seq_len, dm))

    # Calculate the positional encoding
    for pos in range(max_seq_len):
        for i in range(dm):
            if i % 2 == 0:
                PE[pos, i] = np.sin(pos / (10000 ** (i / dm)))
            else:
                PE[pos, i] = np.cos(pos / (10000 ** ((i - 1) / dm)))

    return PE
