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

Template file to specify the parameters required to instruct reaction_base.py to perform a specific set of reactions.
Included in a reaction file are the following:
- `REACTION_CLASS`:
    - The name of the reaction class or set of reactions being specified
    - Ex.: REACTION_CLASS = "hydrogen disassociation reaction"
- `REACTANTS`:
    - A list of all the reactants/reagents involved
    - Ex.: REACTANTS = ["H2", "O2", "Li"]
- `PRODUCTS`:
    - A list of all the products involved
    - Ex.: PRODUCTS = ["H2O", "LiO", "F2"]
- `SOLVENTS`:
    - A list of all the solutes involved (whether they are used or not)
    - Ex.: SOLVENTS = ["H2O", "ethoxyethane"]
- `DESIRED`:
    - A string indicating the desired material (must be present in either the reactants and/or products lists)
    - Ex.: DESIRED = "LiO"
- `Ti`:
    - The intended initial temperature values of the vessel (in Kelvin) in which the reaction(s) is(are) occurring.
    - Ex.: Ti = 293.15
- `Vi`:
    - The intended initial volume values of the vessel (in litres) in which the reaction(s) is(are) occurring.
    - Ex.: Vi = 0.1
- Misc.:
    - Additional parameters of the vessel
    - Ex.: Vmin = 0.05
- `activ_energy_arr`:
    - An n x 1 numpy array containing the activation energies for the n reactions intended to occur.
    - Ex.: activ_energy_arr = np.array([1.0, 2.0, 0.5, 1.5])
- `stoich_coeff_arr`:
    - An m x n numpy array containing the stoichiometric coefficients for each of the m reactions and the n reactants.
    - Ex.: stoich_coeff_arr = np.array([[2.0, 1.0, 4.0], [1.0, 1.0, 1.0], [3.0, 0.0, 0.0], [0.0, 2.0, 2.0]])
- `conc_coeff_arr`:
    - An l x n array containing the concentration coefficients used to determine the changes in concentration for all
    of the l materials and n reactions.
    - Ex.: conc_coeff_arr = np.array([
        [-1.0, -2.0, -3.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [-2.0, -2.0, 0.0, 3.0],
        [1.0, 1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 3.0, 3.0]
    ])
"""

# importing numpy is required to create the necessary arrays
import numpy as np

# name the reaction class
REACTION_CLASS = "SPECIFY THE NAME OF THE REACTION CLASS/SET HERE"

# add the names of the reactants, products, and solutes available to all reactions
REACTANTS = ["ENTER REACTANTS HERE"]
PRODUCTS = ["ENTER PRODUCTS HERE"]
SOLVENTS = ["ENTER SOLVENTS HERE"]

# add the initial thermodynamic values
Ti = 0.0 # in Kelvin
Vi = 0.0 # in Litres

# sample additional vessel parameters
dt = 0.0
Tmin = 0.0
Tmax = 0.0
dT = 0.0
Vmin = 0.0
Vmax = 0.0
dV = 0.0

# add the arrays containing rate calculation parameters; these include:
    # the pre-exponentional factors for each reaction (nx1 array)
    # the activation energies for each reaction (nx1 array)
    # the stoichiometric coefficients (nxm array)
pre_exp_arr = np.array(
    [
        1.0, # pre-exp factor for reaction 0
        1.0, # pre-exp factor for reaction 1
        1.0 # pre-exp factor for reaction 2
    ]
)
activ_energy_arr = np.array(
    [
        0.0, # activation energy for reaction 0
        0.0, # activation energy for reaction 1
        0.0 # activation energy for reaction 2
    ]
)
stoich_coeff_arr = np.array(
    [
        [0.0], # stoichiometric coefficients for reaction 0
        [0.0], # stoichiometric coefficients for reaction 1
        [0.0] # stoichiometric coefficients for reaction 2
    ]
)

# add the array containing concentration change calculations
# this will be an lxn array for the l involved materials and n occurring reactions
conc_coeff_arr = np.array(
    [
        [0.0], # concentration calculation coefficients for material 0
        [0.0], # concentration calculation coefficients for material 1
        [0.0] # concentration calculation coefficients for material 2
    ]
)
