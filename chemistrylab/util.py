import math
import copy
import numpy as np
import chemistrylab


def convert_material_dict_to_volume(material_dict,
                                    # the density of solution does not affect the original volume of solvent
                                    ):
    volume_dict = {}
    total_volume = 0
    for M in material_dict:
        if not material_dict[M][0].is_solute():
            mass = material_dict[M][0].get_molar_mass() * material_dict[M][1]
            volume = mass / material_dict[M][0].get_density()
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
        if math.isclose(material_dict[M][1], 0.0, rel_tol=1e-5):
            new_material_dict.pop(M)
    return new_material_dict


def organize_solute_dict(material_dict,
                         solute_dict):
    for M in material_dict:
        if material_dict[M][0].is_solvent():
            for Solute in solute_dict:
                if M not in solute_dict[Solute]:
                    solute_dict[Solute][M] = 0.0
    new_solute_didct = copy.deepcopy(solute_dict)
    for Solute in solute_dict:
        if Solute not in material_dict:
            new_solute_didct.pop(Solute)
    return new_solute_didct


def check_overflow(material_dict,
                   solute_dict,
                   v_max,
                   ):
    volume_dict, total_volume = convert_material_dict_to_volume(material_dict)
    overflow = total_volume - v_max
    if overflow > 1e-6:
        reward = -10  # punishment
        d_percentage = v_max / total_volume
        for M in material_dict:
            material_dict[M][1] *= d_percentage
        # solute dict
        if solute_dict:
            for Solute in solute_dict:
                for Solvent in solute_dict[Solute]:
                    solute_dict[Solute][Solvent] *= d_percentage

    return material_dict, solute_dict


def generate_sate_old(vessel_list=None,
                      max_n_vessel=5,
                      max_n_material=10,
                      n_layer_bits=100,
                      ):
    # check error
    if len(vessel_list) > max_n_vessel:
        raise ValueError('number of vessels exceeds the max')

    # calculate bits
    n_material_bits = 11  # number of bits for each material in material_dict
    n_solute_bits = 3  # number of bits for each solute-solvent pair in solute dict
    n_total_material_dict_bits = n_material_bits * max_n_material  # total number of bits for material_dict for each vessel
    n_total_solute_dict_bits = (max_n_material - 1) * n_solute_bits * (
            max_n_material - 1)  # total number of bits for solute_dict for each vessel
    n_vessel_bits = n_total_material_dict_bits + n_total_solute_dict_bits + n_layer_bits  # total number of bits in a vessel

    # create state vector with zeros
    state_vector = np.zeros(shape=(n_vessel_bits * max_n_vessel, 1))

    # fill vector
    vessel_counter = 0
    for vessel in vessel_list:
        # gather data:
        material_dict = vessel.get_material_dict()
        solute_dict = vessel.get_solute_dict()

        # check error
        if len(material_dict) > max_n_material:
            raise ValueError('number of material exceeds the max')

        # material_dict:
        material_counter = 0
        for material in material_dict:
            starting_bit = vessel_counter * n_vessel_bits + material_counter * n_material_bits
            state_vector[(starting_bit + 0), 0] = material_dict[material][0].get_index()  # indexed material name
            state_vector[(starting_bit + 1), 0] = material_dict[material][0].get_density()
            state_vector[(starting_bit + 2), 0] = material_dict[material][0].get_polarity()
            state_vector[(starting_bit + 3), 0] = material_dict[material][0].get_temperature()
            state_vector[(starting_bit + 4), 0] = material_dict[material][0].get_pressure()
            # one hot encoding phase
            if material_dict[material][0].get_phase() == 's':
                state_vector[(starting_bit + 5), 0] = 1.0
            else:
                state_vector[(starting_bit + 5), 0] = 0.0
            if material_dict[material][0].get_phase() == 'l':
                state_vector[(starting_bit + 6), 0] = 1.0
            else:
                state_vector[(starting_bit + 6), 0] = 0.0
            if material_dict[material][0].get_phase() == 'g':
                state_vector[(starting_bit + 7), 0] = 1.0
            else:
                state_vector[(starting_bit + 7), 0] = 0.0
            state_vector[(starting_bit + 8), 0] = material_dict[material][0].get_charge()
            state_vector[(starting_bit + 9), 0] = material_dict[material][0].get_molar_mass()
            state_vector[(starting_bit + 10), 0] = material_dict[material][1]  # amount of material
            material_counter += 1

        # solute_dict:
        solute_solvent_pair_counter = 0
        for solute in solute_dict:
            for solvent in solute_dict[solute]:
                starting_bit = vessel_counter * n_vessel_bits + n_total_material_dict_bits + solute_solvent_pair_counter * n_solute_bits
                state_vector[(starting_bit + 0), 0] = material_dict[solute][0].get_index()
                state_vector[(starting_bit + 1), 0] = material_dict[solvent][0].get_index()
                state_vector[(starting_bit + 2), 0] = solute_dict[solute][solvent]
                solute_solvent_pair_counter += 1

        # layer:
        starting_bit = (vessel_counter + 1) * n_vessel_bits
        state_vector[(starting_bit - n_layer_bits):starting_bit, 0] = vessel.get_layers()
        vessel_counter += 1
    return state_vector


def generate_state(vessel_list,
                   max_n_vessel=5):
    # check error
    if len(vessel_list) > max_n_vessel:
        raise ValueError('number of vessels exceeds the max')

    # create a list of the instances of all the subclasses of Material class
    total_num_material = chemistrylab.material.total_num_material
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
        for material in material_dict:
            index = material_dict[material][0].get_index()
            material_dict_matrix[index, 0] = material_dict[material][0].get_density()
            material_dict_matrix[index, 1] = material_dict[material][0].get_polarity()
            material_dict_matrix[index, 2] = material_dict[material][0].get_temperature()
            material_dict_matrix[index, 3] = material_dict[material][0].get_pressure()
            # one hot encoding phase
            if material_dict[material][0].get_phase() == 's':
                material_dict_matrix[index, 4] = 1.0
            else:
                material_dict_matrix[index, 4] = 0.0
            if material_dict[material][0].get_phase() == 'l':
                material_dict_matrix[index, 5] = 1.0
            else:
                material_dict_matrix[index, 5] = 0.0
            if material_dict[material][0].get_phase() == 'g':
                material_dict_matrix[index, 6] = 1.0
            else:
                material_dict_matrix[index, 6] = 0.0
            material_dict_matrix[index, 7] = material_dict[material][0].get_charge()
            material_dict_matrix[index, 8] = material_dict[material][0].get_molar_mass()
            material_dict_matrix[index, 9] = material_dict[material][1]
        # solute_dict:
        for solute in solute_dict:
            solute_index = material_dict[solute][0].get_index()
            for solvent in solute_dict[solute]:
                solvent_index = material_dict[solvent][0].get_index()
                solute_dict_matrix[solute_index, solvent_index] = solute_dict[solute][solvent]
        current_vessel_state = [material_dict_matrix, solute_dict_matrix, layer_vector]
        state.append(current_vessel_state)

    # fill the rest vessel state with zeros
    for i in range(max_n_vessel - len(vessel_list)):
        material_dict_matrix = np.zeros(shape=(total_num_material, 10), dtype=np.float32)
        solute_dict_matrix = np.zeros(shape=(total_num_material, total_num_material), dtype=np.float32)
        air = chemistrylab.material.Air()
        layer_vector = np.zeros(np.shape(state[-1][-1])) + air.get_color()
        state.append([material_dict_matrix, solute_dict_matrix, layer_vector])
    return state
