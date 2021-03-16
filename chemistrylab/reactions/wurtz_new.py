import sys
import numpy as np

sys.path.append("../../")

from chemistrylab.lab.de import De
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

rate_coef = np.array([[2, 0, 0, 1],
                      [1, 1, 0, 1],
                      [1, 0, 1, 1],
                      [0, 2, 0, 1],
                      [0, 1, 1, 1],
                      [0, 0, 2, 1]])

reaction_coef = np.array([[-2, -1, -1, 0, 0, 0],
                          [0, -1, 0, -2, -1, 0],
                          [0, 0, -2, 0, -1, -2],
                          [-2, -2, -2, -2, -2, -2],
                          [1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1],
                          [2, 2, 2, 2, 2, 2]])

wurtz_de = De(rate_coef, np.ones(6), reaction_coef, len(REACTANTS))


class WurtzReaction(_Reaction):
    def __init__(self, initial_materials, initial_solutes, reactants=REACTANTS, products=PRODUCTS, materials=ALL_MATERIALS,
                 solutes=SOLUTES, desired=None, de: De = wurtz_de, solver='RK45', overlap=False, nmax=None, max_mol=2, thresh=1e-8, Ti=0.0,
                 Tmin=0.0, Tmax=0.0, dT=0.0, Vi=0.0, Vmin=0.0, Vmax=0.0, dV=0.0, dt=0):
        super(WurtzReaction, self).__init__(initial_materials, initial_solutes, reactants, products, materials,
                                       de, solver, solutes, desired, overlap, nmax, max_mol, thresh, Ti, Tmin, Tmax, dT,
                                       Vi, Vmin, Vmax, dV, dt)

        self.name = 'wurtz_reaction'

