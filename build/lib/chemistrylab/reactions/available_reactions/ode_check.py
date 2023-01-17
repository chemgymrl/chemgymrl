# reaction of the form A + B -> C; c -> A + B
import numpy as np

# name the reaction class
REACTION_CLASS = "ode check reactions"

# add the names of the reactants, products, and solutes available to all reactions
REACTANTS = [
    "Na",
    "Cl",
    "NaCl",
]
PRODUCTS = [
    "Na",
    "Cl",
    "NaCl",
]
SOLVENTS = ["H2O"]

# define the desired material
DESIRED = "NaCl"

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
    # the activation energies for each reaction (6x1 array)
    # the stoichiometric coefficients (6x4 array)
activ_energy_arr = np.array(
    [
        1000, # activation energy for reaction 0
        1, # activation energy for reaction 1
    ]
)

# Note: R0 = Na, R1 = Cl, R2 = NaCl
stoich_coeff_arr = np.array(
    [ #  R0   R1   R2
        [1.0, 1.0, 0.0], # stoichiometric coefficients for reaction 0
        [0.0, 0.0, 1.0], # stoichiometric coefficients for reaction 1
    ]
)

# add the array containing concentration change calculations
# this will be an 11x6 array for the 11 involved materials and 6 occurring reactions
conc_coeff_arr = np.array(
    [ #  r0    r1
        [-1.0, 1.0], # concentration calculation coefficients for Na
        [-1.0, 1.0], # concentration calculation coefficients for Cl
        [1.0, -1.0], # concentration calculation coefficients for NaCl

    ]
)
