import math
import copy
import numpy as np
import sys

sys.path.append("../../")
from chemistrylab.chem_algorithms import material


def convert_material_dict_to_volume(material_dict,
                                    # the density of solution does not affect the original volume of solvent
                                    ):
    volume_dict = {}
    total_volume = 0
    for M in material_dict:
        if not material_dict[M][0]().is_solute():
            # get the mass (in grams) of the material using its molar mass (in grams/mol)
            mass = material_dict[M][0]().get_molar_mass() * material_dict[M][1]

            # get the volume (in cubic centimeters) of the material using its
            # mass (in grams) and density (in grams per cubic centimeter)
            volume = mass / material_dict[M][0]().get_density()

            # convert volume in cm**3 to L
            volume = volume / 1000

            volume_dict[M] = volume
            total_volume += volume

    return volume_dict, total_volume


def convert_volume_to_mole(volume,
                           density,
                           molar_mass,
                           ):
    mass = volume * density
    mole = mass / molar_mass
    return mole


def organize_material_dict(material_dict):
    new_material_dict = copy.deepcopy(material_dict)
    for M in material_dict:
        if math.isclose(material_dict[M][1], 0.0, rel_tol=1e-6):  # remove material with 0.0 amount
            new_material_dict.pop(M)
    return new_material_dict


def organize_solute_dict(material_dict,
                         solute_dict):
    for M in material_dict:
        if material_dict[M][0]().is_solvent():  # if the material is solvent add to each solute with 0.0 amount
            for Solute in solute_dict:
                if M not in solute_dict[Solute]:
                    solute_dict[Solute][M] = 0.0
    new_solute_dict = copy.deepcopy(solute_dict)
    for Solute in solute_dict:  # remove solute if it's not in material dict (avoid Solute: [Solvent, 0.0] )
        if Solute not in material_dict:
            new_solute_dict.pop(Solute)
    return new_solute_dict


def check_overflow(material_dict,
                   solute_dict,
                   v_max,
                   ):
    __, total_volume = convert_material_dict_to_volume(material_dict)  # convert from mole to L
    overflow = total_volume - v_max  # calculate overflow
    reward = 0  # default 0 if no overflow
    if overflow > 1e-6:  # if overflow
        reward = -1  # punishment
        d_percentage = v_max / total_volume  # calculate the percentage of material left
        for M in material_dict:
            material_dict[M][1] *= d_percentage
        # solute dict
        if solute_dict:  # if not empty
            for Solute in solute_dict:
                for Solvent in solute_dict[Solute]:
                    solute_dict[Solute][Solvent] *= d_percentage  # update the solute_dict based on percentage

    return material_dict, solute_dict, reward


def generate_state(vessel_list,
                   max_n_vessel=5):
    all_materials = [[mat().get_name() for mat in material.Material.__subclasses__()], material.Material.__subclasses__()]
    # check error
    if len(vessel_list) > max_n_vessel:
        raise ValueError('number of vessels exceeds the max')

    # create a list of the instances of all the subclasses of Material class
    total_num_material = material.total_num_material
    state = []
    for vessel in vessel_list:
        # initialize three matrix
        material_dict_matrix = np.zeros(shape=(total_num_material, 10), dtype=np.float32)
        solute_dict_matrix = np.zeros(shape=(total_num_material, total_num_material), dtype=np.float32)
        layer_vector = vessel.get_layers()

        # gather data:
        material_dict = vessel.get_material_dict()
        solute_dict = vessel.get_solute_dict()

        # fill material_dict_matrix and solute_dict_matrix

        # material_dict:
        for materials in material_dict:
            index = material_dict[materials][0]().get_index()
            material_dict_matrix[index, 0] = material_dict[materials][0]().get_density()
            material_dict_matrix[index, 1] = material_dict[materials][0]().get_polarity()
            material_dict_matrix[index, 2] = material_dict[materials][0]().get_temperature()
            material_dict_matrix[index, 3] = material_dict[materials][0]().get_pressure()
            # one hot encoding phase
            if material_dict[materials][0]().get_phase() == 's':
                material_dict_matrix[index, 4] = 1.0
            else:
                material_dict_matrix[index, 4] = 0.0
            if material_dict[materials][0]().get_phase() == 'l':
                material_dict_matrix[index, 5] = 1.0
            else:
                material_dict_matrix[index, 5] = 0.0
            if material_dict[materials][0]().get_phase() == 'g':
                material_dict_matrix[index, 6] = 1.0
            else:
                material_dict_matrix[index, 6] = 0.0
            material_dict_matrix[index, 7] = material_dict[materials][0]().get_charge()
            material_dict_matrix[index, 8] = material_dict[materials][0]().get_molar_mass()
            material_dict_matrix[index, 9] = material_dict[materials][1]

        # solute_dict:
        for solute in solute_dict:
            solute_class = all_materials[1][all_materials[0].index(solute)]()
            solute_index = solute_class.get_index()
            for solvent in solute_dict[solute]:
                solvent_index = all_materials[1][all_materials[0].index(solvent)]().get_index()
                solute_dict_matrix[solute_index, solvent_index] = solute_dict[solute][solvent]

        current_vessel_state = [material_dict_matrix, solute_dict_matrix, layer_vector]
        state.append(current_vessel_state)

    # fill the rest vessel state with zeros
    for __ in range(max_n_vessel - len(vessel_list)):
        material_dict_matrix = np.zeros(shape=(total_num_material, 10), dtype=np.float32)
        solute_dict_matrix = np.zeros(shape=(total_num_material, total_num_material), dtype=np.float32)
        air = material.Air()
        layer_vector = np.zeros(np.shape(state[-1][-1])) + air.get_color()
        state.append([material_dict_matrix, solute_dict_matrix, layer_vector])
    return state
