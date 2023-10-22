#!/usr/bin/env python3

import numpy as np

class MultiNormal:
    def __init__(self, data):
        """
        Initialize the MultiNormal class with data to calculate mean and covariance.

        Args:
            data (numpy.ndarray): A 2D numpy array of shape (d, n) containing the data set.
                n is the number of data points, and d is the number of dimensions in each data point.

        Raises:
            TypeError: If data is not a 2D numpy array.
            ValueError: If n is less than 2, indicating that data must contain multiple data points.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.dot(data - self.mean, (data - self.mean).T) / (data.shape[1] - 1)

    def pdf(self, x):
        """
        Calculate the Probability Density Function (PDF) at a data point.

        Args:
            x (numpy.ndarray): A 2D numpy array of shape (d, 1) containing the data point whose PDF should be calculated.
                d is the number of dimensions of the Multinomial instance.

        Returns:
            pdf_value: The value of the PDF.

        Raises:
            TypeError: If x is not a numpy.ndarray or if x does not have the expected shape.
        """
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape != self.mean.shape:
            raise ValueError(f"x must have the shape {self.mean.shape}")

        d = self.mean.shape[0]
        cov_inv = np.linalg.inv(self.cov)
        factor = 1 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov)))
        exponent = -0.5 * np.dot(np.dot((x - self.mean).T, cov_inv), x - self.mean)
        pdf_value = factor * np.exp(exponent)

        return pdf_value
