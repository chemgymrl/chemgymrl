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


def organize_solute_dict(material_dict,
                         solute_dict):
    for M in material_dict:
        if material_dict[M][0].is_solvent():
            for Solute in solute_dict:
                if M not in solute_dict[Solute]:
                    solute_dict[Solute][M] = 0.0
    return solute_dict


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
        if solute_dict is not None:
            for Solute in solute_dict:
                for Solvent in solute_dict[Solute]:
                    solute_dict[Solute][Solvent] *= d_percentage

    return material_dict, solute_dict
