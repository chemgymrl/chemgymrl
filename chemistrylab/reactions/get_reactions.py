'''
Module to provide methods to assist or access reactions
'''

from glob import glob
import os
import string
import sys

# allow this module to access the chemistrylab directory
sys.path.append("../../")
from chemistrylab.chem_algorithms.material import get_materials

current_file = __file__

def convert_to_class(materials=None):
    '''
    Function to convert a material's string representation to it's class representation.
    '''

    (material_names, material_objects) = get_materials()

    all_material_classes = []

    # iterate through each material in the list of provided material strings
    for material in materials:
        # get the index of the current material in the list of names from get_materials;
        # obtain the class representation of the material using the index
        material_class = None
        for i, material_name in enumerate(material_names):
            if any([
                material_name == material,
                material_objects[i]().get_name() == material
            ]):
                material_class = material_objects[i]

        # append the class representation of the current material to a list
        all_material_classes.append(material_class)

    return all_material_classes

def get_reactions():
    '''
    Function to obtain the Reaction classes of each reaction in the `reactions` directory.
    '''

    # find the path to the reactions directory
    all_reaction_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py")

    reactions = []
    for reaction_file in glob(all_reaction_files):
        # obtain the name of each file in the reactions directory
        name = os.path.splitext(os.path.basename(reaction_file))[0]

        # only files that are named similar to reaction_1.py are valid reaction files
        if all([
            name[0:8] == "reaction",
            all([item in string.digits for item in list(name.split("_")[-1])])
        ]):
            # import the reaction file, extract the Reaction class, and add the class to a list
            module = __import__(name)
            r_class = module.Reaction
            reactions.append(r_class)

    return reactions
