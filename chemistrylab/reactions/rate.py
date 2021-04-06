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


class WurtzRates:
    def __init__(self):
        pass
    
    def get_rates(self, k, conc):
        rate = np.zeros(6)
        rate[0] = k[0] * (conc[0] ** 2) * (conc[1] ** 0) * (conc[2] ** 0) * (conc[3] ** 1)
        rate[1] = k[1] * (conc[0] ** 1) * (conc[1] ** 1) * (conc[2] ** 0) * (conc[3] ** 1)
        rate[2] = k[2] * (conc[0] ** 1) * (conc[1] ** 0) * (conc[2] ** 1) * (conc[3] ** 1)
        rate[3] = k[3] * (conc[0] ** 0) * (conc[1] ** 2) * (conc[2] ** 0) * (conc[3] ** 1)
        rate[4] = k[4] * (conc[0] ** 0) * (conc[1] ** 1) * (conc[2] ** 1) * (conc[3] ** 1)
        rate[5] = k[5] * (conc[0] ** 0) * (conc[1] ** 0) * (conc[2] ** 2) * (conc[3] ** 1)
        return rate


class Rates:
    def __init__(self, exps):
        rates_fn = []
        for exp in exps:
            rates_fn.append(ExponentialRate(exp))
        self.rates_fn = rates_fn

    def get_rates(self, k, conc):
        rates = np.zeros(len(self.rates_fn))
        for i, rate in enumerate(self.rates_fn):
            rates[i] = rate.run(k[i], conc)

        return rates
