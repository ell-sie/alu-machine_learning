#!/usr/bin/env python3
"""A class Exponential representing an Exponential Distribution.
The exponential distribution is a continuous probability distribution that 
models the time between events in a Poisson process. 
It is commonly used to describe the distribution of waiting times or inter-arrival 
times in various real-world scenarios, such as the time between arrivals of customers 
at a service center, the time between radioactive decay events, 
or the lifespan of electronic components."""


class Exponential:
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = float((1 / sum(data)) / (1 / len(data)))

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""

        e = 2.7182818285
        if x < 0:
            return (0)
        return self.lambtha * e ** ((-1 * self.lambtha) * x)

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""
        # F(x) = 1 - e^(-λx), for x ≥ 0

        e = 2.7182818285
        if x < 0:
            return 0
        return 1 - e ** (-self.lambtha * x)