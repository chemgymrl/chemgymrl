import sys
import numpy as np

sys.path.append("../../")

from chemistrylab.reactions.rate import Rates
from chemistrylab.reactions.reaction_base import _Reaction

REACTANTS = [
    "1-chlorohexane",
    "2-chlorohexane",
    "3-chlorohexane",
    "Na"
]
PRODUCTS = [
    "dodecane",
    "5-methylundecane",
    "4-ethyldecane",
    "5,6-dimethyldecane",
    "4-ethyl-5-methylnonane",
    "4,5-diethyloctane",
    "NaCl"
]
ALL_MATERIALS = REACTANTS + PRODUCTS
SOLUTES = ["H2O"]

EXP_COEF = np.ones(6)

class Reaction(_Reaction):
    def __init__(self, initial_materials, initial_solutes, desired, rate_fn: Rates, solutes=SOLUTES,
                 reactants=REACTANTS, products=PRODUCTS, materials=ALL_MATERIALS, exp_coef=EXP_COEF, overlap=False,
                 nmax=None, max_mol=2, thresh=1e-8, Ti=0.0, Tmin=0.0, Tmax=0.0, dT=0.0, Vi=0.0, Vmin=0.0, Vmax=0.0,
                 dV=0.0, dt=0):
        super(Reaction, self).__init__(initial_materials, initial_solutes, reactants, products, materials, desired, exp_coef, rate_fn, solutes,
                 overlap, nmax, max_mol, thresh, Ti, Tmin, Tmax, dT, Vi, Vmin, Vmax, dV, dt)

        self.name = 'wurtz_reaction'

    def get_conc_change(self, rates, conc, dt):
        dC = np.zeros(11)
        dC[0] = ((-2.0 * rates[0]) + (-1.0 * rates[1]) + (-1.0 * rates[2])) * dt  # change in 1-chlorohexane
        dC[1] = ((-1.0 * rates[1]) + (-2.0 * rates[3]) + (-1.0 * rates[4])) * dt  # change in 2-chlorohexane
        dC[2] = ((-2.0 * rates[2]) + (-1.0 * rates[4]) + (-2.0 * rates[5])) * dt  # change in 3-chlorohexane
        dC[3] = -2.0 * (rates[0] + rates[1] + rates[2] + rates[3] + rates[4] + rates[5]) * dt  # change in Na
        dC[4] = 1.0 * rates[0] * dt  # change in dodecane
        dC[5] = 1.0 * rates[1] * dt  # change in 5-methylundecane
        dC[6] = 1.0 * rates[2] * dt  # change in 4-ethyldecane
        dC[7] = 1.0 * rates[3] * dt  # change in 5,6-dimethyldecane
        dC[8] = 1.0 * rates[4] * dt  # change in 4-ethyl-5-methylnonane
        dC[9] = 1.0 * rates[5] * dt  # change in 4,5-diethyloctane
        dC[10] = 2.0 * (rates[0] + rates[1] + rates[2] + rates[3] + rates[4] + rates[5]) * dt  # change in NaCl

        # ensure the changes in reactant concentration do not exceed the concentrations available
        dC[0] = np.max([dC[0], -1.0 * conc[0]])
        dC[1] = np.max([dC[1], -1.0 * conc[1]])
        dC[2] = np.max([dC[2], -1.0 * conc[2]])
        dC[3] = np.max([dC[3], -1.0 * conc[3]])
        return dC


