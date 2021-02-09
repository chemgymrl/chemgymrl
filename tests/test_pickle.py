import pickle
import os
import sys
sys.path.append('../')
sys.path.append('../chemistrylab/reactions')

# ask user if they want to see stats of reaction vessel
show_stats = input("Show Reaction Vessel Stats ('Y'/'N') >>> ")

if show_stats.lower() in ["y", "yes"]:
    # open and check the material dict
    vessel_path = os.path.join(os.getcwd(), "extract_vessel.pickle")
    print(vessel_path)
    with open(vessel_path, 'rb') as open_file:
        v = pickle.load(open_file)

    print("")
    print("---------- VESSEL ----------")
    print("Label: {}".format(v.label))

    print("")
    print("---------- THERMODYNAMIC VARIABLES ----------")
    print("Temperature (in K): {:e}".format(v.temperature))
    print("Volume (in L): {:e}".format(v.volume))
    print("Pressure (in kPa): {:e}".format(v.pressure))

    print("")
    print("---------- MATERIAL_DICT ----------")
    for material, value_list in v._material_dict.items():
        print("{} : {}".format(material, value_list))

    print("")
    print("---------- SOLUTE_DICT ----------")
    for solute, value_list in v._solute_dict.items():
        print("{} : {}".format(solute, value_list))

