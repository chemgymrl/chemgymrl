'''
File to specify the parameters required to instruct reaction_base.py to perform the
chloro-wurtz reactions
'''

import numpy as np

# name the reaction class
REACTION_CLASS = "chloro-wurtz reactions"

# add the names of the reactants, products, and solutes available to all reactions
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
SOLUTES = ["H2O"]

# define the desired material
DESIRED = "dodecane"

# add the initial thermodynamic values
Ti = 297.0 # in Kelvin
Vi = 0.002 # in Litres

# additional vessel parameters
dt = 0.01
Tmin = 250.0
Tmax = 500.0
dT = 50.0
Vmin = 0.001
Vmax = 0.005
dV = 0.0005

# add the arrays containing rate calculation parameters; these include:
    # the activation energies for each reaction (6x1 array)
    # the stoichiometric coefficients (6x4 array)
activ_energy_arr = np.array(
    [
        [1.0], # activation energy for reaction 0
        [1.0], # activation energy for reaction 1
        [1.0], # activation energy for reaction 2
        [1.0], # activation energy for reaction 3
        [1.0], # activation energy for reaction 4
        [1.0]  # activation energy for reaction 5
    ]
)

# Note: R0 = 1-chlorohexane, R1 = 2-chlorohexane, R2 = 3-chlorohexane, R3 = Na
stoich_coeff_arr = np.array(
    [ #  R0   R1   R2   R3
        [2.0, 0.0, 0.0, 1.0], # stoichiometric coefficients for reaction 0
        [1.0, 1.0, 0.0, 1.0], # stoichiometric coefficients for reaction 1
        [1.0, 0.0, 1.0, 1.0], # stoichiometric coefficients for reaction 2
        [0.0, 2.0, 0.0, 1.0], # stoichiometric coefficients for reaction 3
        [0.0, 1.0, 1.0, 1.0], # stoichiometric coefficients for reaction 4
        [0.0, 0.0, 2.0, 1.0]  # stoichiometric coefficients for reaction 5
    ]
)

# add the array containing concentration change calculations
# this will be an 11x6 array for the 11 involved materials and 6 occurring reactions
conc_coeff_arr = np.array(
    [ #  r0    r1    r2    r3   r4   r5
        [-2.0, -1.0, -1.0, 0.0, 0.0, 0.0], # concentration calculation coefficients for 1-chlorohexane
        [0.0, -1.0, 0.0, -2.0, -1.0, 0.0], # concentration calculation coefficients for 2-chlorohexane
        [0.0, 0.0, -1.0, 0.0, -1.0, -2.0], # concentration calculation coefficients for 3-chlorohexane
        [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0], # concentration calculation coefficients for Na
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], # concentration calculation coefficients for dodecane
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], # concentration calculation coefficients for 5-methylundecane
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # concentration calculation coefficients for 4-ethyldecane
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # concentration calculation coefficients for 5,6-dimethyldecane
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # concentration calculation coefficients for 4-ethyl-5-methylnonane
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # concentration calculation coefficients for 4,5-diethyloctane
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0] # concentration calculation coefficients for NaCl
    ]
)
