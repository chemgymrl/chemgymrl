"""
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.

File to specify the parameters required to instruct reaction_base.py to perform the
chloro-wurtz reactions
"""

import numpy as np

# name the reaction class
REACTION_CLASS = "decomp reactions"

# add the names of the reactants, products, and solutes available to all reactions
REACTANTS = [
    "NaCl"
]
PRODUCTS = [
    "Na",
    "Cl"
]
SOLVENTS = ["H2O"]

# add the initial thermodynamic values
Ti = 297.0 # in Kelvin
Vi = 1 # in Litres

# additional vessel parameters
dt = 0.01
Tmin = 250.0
Tmax = 500.0
dT = 50.0
Vmin = 0.001
Vmax = 2
dV = 0.005


# add the arrays containing rate calculation parameters; these include:
    # the pre-exponentional factors for each reaction (1x1 array)
    # the activation energies for each reaction (1x1 array)
    # the stoichiometric coefficients (1x1 array)
pre_exp_arr = np.array(
    [
        1.0, # pre-exp factor for reaction 0
    ]
)
activ_energy_arr = np.array(
    [
        8.0 # activation energy for reaction 0
    ]
)

# Note: R0 = NaCl
stoich_coeff_arr = np.array(
    [ #  R0
        [1.0] # stoichiometric coefficients
    ]
)

# add the array containing concentration change calculations
# this will be an 3x1 array for the 3 involved materials and 1 occurring reaction
conc_coeff_arr = np.array(
    [ #  r0
        [-1.0], # concentration calculation coefficients for NaCl
        [1.0], # concentration calculation coefficients for Na
        [1.0], # concentration calculation coefficients for Cl
        [0.0] # concentration calculation coefficients for H2O
    ]
)
