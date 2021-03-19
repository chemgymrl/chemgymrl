'''
A file including valuable information about the capabilities of each environment.
'''

import itertools
import sys

# ensure all necessary modules can be found
sys.path.append("../") # to access chemistrylab
sys.path.append("../chemistrylab/reactions") # to access all reactions

# import all local modules
import chemistrylab

# import the materials file
from chemistrylab.chem_algorithms import material

# take the materials from the material.py file
(materials, mat_objects) = material.get_materials()

# -------------------- # INPUT MATERIALS # -------------------- #

# define the chloro hydrocarbons available
all_chloro = [material for material in materials if "chloro" in material.lower()]

# reaction_bench requires at least one chloro hydrocarbon and Na to generate a Wurtz reaction;
# additionally, a single solute must be used as well;
# all possible valid configurations are generated and tabulated below

available_input_materials = []
for i, __ in enumerate(all_chloro):
    # create a list of every combination of available chloro-hydrocarbons
    config_list = list(itertools.combinations(all_chloro, i+1))
    new_configs = [old_config + ("Na",) for old_config in config_list]
    available_input_materials += new_configs

# -------------------- # SOLUTES # -------------------- #

# define the solutes available'
# they are noticeable by the 'is_solute' attribute
available_solutes = [
    material for i, material in enumerate(materials) if mat_objects[i]().is_solute()
]

# -------------------- # DESIRED MATERIALS # -------------------- #

# define a list to contain all the available desired output materials
available_desired_materials = []

# flatten the list of available input materials
flat_input_list = [material for config in available_input_materials for material in config]

for i, material in enumerate(materials):
    if all([
        # only materials with a defined enthalpy of fusion and enthalpy;
        # of vapour can be tracked through all the environments
        mat_objects[i]()._enthalpy_fusion, # need a defined enthalpy of fusion
        mat_objects[i]()._enthalpy_vapor, # need a defined enthalpy of vapor
        material not in flat_input_list, # the hydrocarbon is not an input material
        not mat_objects[i]().is_solute() # the hydrocarbon is not a solute
    ]):
        available_desired_materials.append(material)


print(available_input_materials)
print(available_solutes)
print(available_desired_materials)
