#!/usr/bin/env python3

import numpy as np
from scipy.special import comb

def binomial_distribution(x, n, P):
    """
    Calculate the binomial distribution given the number of successes (x), total trials (n), and success probability (P).
    
    Parameters:
        x (int): The number of successes.
        n (int): The total number of trials.
        P (numpy.ndarray): A 1D numpy array containing success probabilities.

    Returns:
        numpy.ndarray: A 1D numpy array containing the probability of obtaining x successes for each probability in P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x >= 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not all(0 <= p <= 1 for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    binomial_coefficient = comb(n, x)
    binomial_probabilities = binomial_coefficient * P**x * (1 - P)**(n - x)

    return binomial_probabilities

if __name__ == '__main':
    P = np.linspace(0, 1, 11)
    print(binomial_distribution(26, 130, P))
