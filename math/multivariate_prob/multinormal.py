#!/usr/bin/env python3
"""
        Initialize the MultiNormal class with data to calculate mean and covariance.

        Args:
            data (numpy.ndarray): A 2D numpy array of shape
            (d, n) containing the data set.
                n is the number of data points, and d is
                the number of dimensions in each data point.
"""
import numpy as np


class MultiNormal:
    def __init__(self, data):
        """
        Raises:
            TypeError: If data is not a 2D numpy array.
            ValueError: If n is less than 2, indicating
            that data must contain multiple data points.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, axis=1, keepdims=True)
        cov = np.matmul(data - mean, data.T - mean.T) / (data.shape[1] - 1)
        self.mean = mean
        self.cov = cov

    def pdf(self, x):
        """
        Calculate the Probability Density Function (PDF) at a data point.

        Args:
            x (numpy.ndarray): A 2D numpy array of shape (d, 1)
            containing the data point whose PDF should be calculated.
                d is the number of dimensions of the Multinomial instance.

        Returns:
            pdf_value: The value of the PDF.

        Raises:
            TypeError: If x is not a numpy.ndarray or if x does
            not have the expected shape.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (self.cov.shape[0], 1):
            e_msg = "x must have the shape ({}, 1)".format(self.cov.shape[0])
            raise ValueError(e_msg)

        denominator = np.sqrt(((2 * np.pi) ** x.shape[0])
                              * np.linalg.det(self.cov))
        exponent = -0.5 * np.matmul(np.matmul((x - self.mean).T,
                                    np.linalg.inv(self.cov)), x - self.mean)

        return (1 / denominator) * np.exp(exponent[0][0])
