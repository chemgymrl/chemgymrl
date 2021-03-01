import numpy as np


def _exponential_rate(k, conc, exponents):
    rtn = k
    for exp, c in zip(exponents, conc):
        rtn *= c ** exp
    return rtn


class ExponentialRate:
    def __init__(self, exponents):
        self.exponents = exponents

    def run(self, k, conc):
        _exponential_rate(k, conc, self.exponents)


class CustomRate:
    def __init__(self):
        pass

    def run(self, k, conc):
        # do what you want in here
        pass


class Rates:
    def __init__(self, rates_fn: list):
        self.rates_fn = rates_fn

    def get_rates(self, k, conc):
        rates = np.zeros(len(self.rates_fn))
        for i, rate in enumerate(self.rates_fn):
            rates[i] = rate.run(k[i], conc)

        return rates
