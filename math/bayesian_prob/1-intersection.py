#!/usr/bin/env python3

"""
Exceptions:
#         ValueError:
#             - If n is not a positive integer.
#             - If x is not a non-negative integer.
#             - If x is greater than n.
#             - If any value in P or Pr is not in the range [0, 1].
#             - If Pr does not sum to 1.
#         TypeError:
#             - If P or Pr is not a 1D numpy array.
#             - If Pr does not have the same shape as P.
"""
import numpy as np
from scipy.special import comb


def intersection(x, n, P, Pr):
    """
      Calculate the intersection of obtaining certain data
     with various hypothetical probabilities, considering prior beliefs.

     Parameters:
         x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy array containing the
         various hypothetical probabilities of developing severe side effects.
        Pr (numpy.ndarray): A 1D numpy array containing
         the prior beliefs of the probabilities in P.

    Returns:
    numpy.ndarray: A 1D numpy array containing the intersection
    of obtaining (x, n) with each probability in P,
     considering the prior beliefs in Pr.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
        "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1 or Pr.shape != P.shape:
        raise TypeError(
        "Pr must be a numpy.ndarray with the same shape as P")
    if any(p < 0 or p > 1 for p in P):
        raise ValueError(
        "All values in P must be in the range [0, 1]")
    if any(p < 0 or p > 1 for p in Pr):
        raise ValueError(
        "All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    likelihoods = [comb(n, x) * (p ** x) * ((1 - p) ** (n - x)) for p in P]
    intersection = likelihoods * Pr

    return intersection
