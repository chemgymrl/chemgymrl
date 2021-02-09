import math
import copy
import numpy as np
import sys

sys.path.append("../../")
from chemistrylab.chem_algorithms import material


MASS_TABLE = {'kg': 1000,
              'g': 1,
              'mg': 1e-3}

VOLUME_TABLE = {'l': 1000,
                  'dl': 100,
                  'ml': 1,
                  'ul': 1/1000,
                  'nl': 1e-6,
                }

def convert_material_dict_to_volume(material_dict,
                                    # the density of solution does not affect the original volume of solvent
                                    ):
    volume_dict = {}
    total_volume = 0
    for M in material_dict:
        if not material_dict[M][0]().is_solute():
            mass = material_dict[M][0]().get_molar_mass() * material_dict[M][1]
            volume = mass / material_dict[M][0]().get_density() # in L
            volume_dict[M] = volume
            total_volume += volume
    print(volume_dict)
    print(total_volume)
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
                    solute_dict[Solute][M] = [0.0, 'mol']
    new_solute_dict = copy.deepcopy(solute_dict)
    for Solute in solute_dict:  # remove solute if it's not in material dict (avoid Solute: [Solvent, 0.0] )
        if Solute not in material_dict:
            new_solute_dict.pop(Solute)
    return new_solute_dict


def check_overflow(material_dict,
                   solute_dict,
                   v_max,
                   ):
    __, total_volume = convert_material_dict_to_volume(material_dict)  # convert from mole to ml
    print(v_max)
    overflow = total_volume - v_max  # calculate overflow
    reward = 0  # default 0 if no overflow
    if overflow > 1e-6:  # if overflow
        reward = -10  # punishment
        d_percentage = v_max / total_volume  # calculate the percentage of material left
        for M in material_dict:
            material_dict[M][1] *= d_percentage
        # solute dict
        if solute_dict:  # if not empty
            for Solute in solute_dict:
                for Solvent in solute_dict[Solute]:
                    solute_dict[Solute][Solvent][0] *= d_percentage  # update the solute_dict based on percentage

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
                solute_dict_matrix[solute_index, solvent_index] = solute_dict[solute][solvent][0]

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


def convert_volume(volume, unit):
    if unit not in VOLUME_TABLE:
        raise KeyError('unit is not in unit conversion table please check your spelling or add the unit to the table')
    return volume * VOLUME_TABLE[unit]


def convert_mass_to_mol(mat, mass, unit='g'):

    if type(mat) == material.Material:
        return mat.molar_mass * MASS_TABLE[unit] * mass
    elif type(mat) == str:
        found = False
        materials = material.get_materials()
        for i, name in enumerate(materials[0]):
            if name == mat:
                found = True
                return materials[1][i].molar_mass * MASS_TABLE[unit] * mass
        if not found:
            raise ValueError('Material name not found')

    else:
        raise TypeError('the material input must be a material class or a string name for the material')


def convert_volume_to_mol(mat, volume, unit='ml'):
    if type(mat) == material.Material:
        mass = convert_volume(volume, unit) * mat.density
        return mass / mat.molar_mass
    elif type(mat) == str:
        found = False
        materials = material.get_materials()
        for i, name in enumerate(materials[0]):
            if name == mat:
                found = True
                mass = convert_volume(volume, unit) * materials[1][i].density
                return mass / materials[1][i].molar_mass

        if not found:
            raise ValueError('Material name not found')

    else:
        raise TypeError('the material input must be a material class or a string name for the material')


def convert_unit_to_mol(mat, measurement, unit):
    if unit in MASS_TABLE.keys():
        return convert_mass_to_mol(mat, measurement, unit)
    elif unit in VOLUME_TABLE.keys():
        return convert_volume_to_mol(mat, measurement, unit)
    elif unit == 'mol':
        return measurement
    else:
        raise ValueError('the specified unit is either not convertible or not in any of our conversion tables')


def convert_material_dict_units(material_dict):
    new_material_dict = {}
    for key, item in material_dict.items():
        unit = 'mol'
        if len(item) == 3:
            print(key, item)
            unit = item[2]
        new_material_dict[key] = [item[0], convert_unit_to_mol(item[0], item[1], unit), 'mol']
    return new_material_dict


def convert_solute_dict_units(solute_dict):
    new_solute_dict = {}
    for solute, solvents in solute_dict.items():
        new_solute_dict[solute] = {}
        for solvent, item in solvents.items():
            unit = 'mol'
            if len(item) == 2:
                print(solvent, item)
                unit = item[2]
                unit = item[1]
            new_solute_dict[solute][solvent] = [convert_unit_to_mol(solvent, item[0], unit)]
    return new_solute_dict
