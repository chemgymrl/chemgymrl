import numpy as np


class De:
    def __init__(self, rate_coef: np.array, exp_coef:np.array, reaction_coef: np.array, num_reagents: int):
        self.rate_coef = rate_coef
        self.exp_coef = exp_coef
        self.reaction_coef = reaction_coef
        self.num_reagents = num_reagents

    def __call__(self, t, y):
        conc = y[:, 0]
        temp = y[:, 1][0]
        conc_change = self.run(conc, temp)
        y[:, 0] = conc_change
        return y

    def run(self, conc, temp):
        k = self.get_reaction_constants(temp, conc)
        rates = self.get_rates(k, conc)
        conc_change = self.get_conc_change(rates)
        conc_change = self.conc_limit(conc_change, conc)
        return conc_change

    def get_rates(self, k, conc):
        rate = conc[:self.num_reagents] ** self.rate_coef
        rate = k * np.array([np.product(x) for x in rate])
        return rate

    def get_reaction_constants(self, temp, conc):
        R = 8.314462619

        agg_conc = 1
        reactant_conc = [conc[i] for i in range(self.num_reagents)]
        non_zero_conc = [_conc for _conc in reactant_conc if _conc != 0.0]
        for conc in non_zero_conc:
            agg_conc *= conc
        exponent = 2 / len(non_zero_conc) if len(non_zero_conc) != 0 else 1
        scaling_factor = (abs(agg_conc)) ** exponent

        const_coef = 1 / scaling_factor
        k = const_coef * np.exp((-1 * self.exp_coef) / (R * temp))
        return k

    def get_conc_change(self, rates):
        dC = np.array([np.sum(x) for x in self.reaction_coef * rates])
        return dC

    def conc_limit(self, conc_change, conc):
        for i in range(self.num_reagents):
            conc_change[i] = np.max([conc_change[i], -1*conc[i]])
        return conc_change

