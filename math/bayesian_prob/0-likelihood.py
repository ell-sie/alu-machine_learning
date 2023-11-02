#!/usr/bin/env python3
import numpy as np
"""
    Calculate the likelihood of obtaining certain
     data given various hypothetical probabilities.

    Parameters:
        x (int): The number of patients that develop
         severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy array containing the various
         hypothetical probabilitiesof developing severe side effects.
"""

def likelihood(x, n, P):
    """


    Returns:
        numpy.ndarray: A 1D numpy array containing the
        likelihood of obtaining the data (x, n) for each probability in P.

    Exceptions:
        ValueError:
            - If n is not a positive integer.
            - If x is not a non-negative integer.
            - If x is greater than n.
            - If any value in P is not in the range [0, 1].
        TypeError:
            - If P is not a 1D numpy array.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x > 0:
        raise ValueError
    ("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not all(0 <= p <= 1 for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    binomial_coefficient = np.math.comb(n, x)
    likelihoods = binomial_coefficient * P**x * (1 - P)**(n - x)

    return likelihoods
