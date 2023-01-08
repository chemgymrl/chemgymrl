import numpy as np

# name the reaction class
REACTION_CLASS = "methyl-red reactions"

# add the names of the reactants, products, and solutes available to all reactions
REACTANTS = []
PRODUCTS = [
    "H",
    "Cl",
    "methyl red"
]
SOLVENTS = ["H2O"]

# add the initial thermodynamic values
Ti = 297.0 # in Kelvin
Vi = 1 # in Litres

# additional vessel parameters
dt = 0.05
Tmin = 250.0
Tmax = 500.0
dT = 50.0
Vmin = 0.001
Vmax = 2
dV = 0.005



activ_energy_arr = np.zeros(1)

stoich_coeff_arr = np.zeros(1)

conc_coeff_arr = np.zeros(1)
