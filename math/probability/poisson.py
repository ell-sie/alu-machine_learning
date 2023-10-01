#!/usr/bin/env python3
"""A class Poisson representing a Poisson Distribution.
The Poisson distribution is a probability distribution that describes 
the number of events that occur within a fixed interval of time or space.
e.g. number of emails received in an hour, 
the number of accidents at an intersection in a day, or 
the number of phone calls to a customer service center in a minute."""


class Poisson():
    """Poisson Distribution"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)) / len(data)

    def pmf(self, k):
        """calculates the value of the PMF for a given number of “successes”"""

        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)

        # calculates
        def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n - 1)

        e = 2.7182818285
        return ((self.lambtha ** k) * (1 / (e ** self.lambtha))) / factorial(k)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""

        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        else:
            res = 0
            for i in range(k + 1):
                res += self.pmf(i)
            return res