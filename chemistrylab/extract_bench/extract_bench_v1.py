'''
Module to access and execute the ExtractWorld Engine

:title: extractworld_v1.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-06-24
'''

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

# import external modules
import os
import pickle
import sys

# import local modules
sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.chem_algorithms import material, util, vessel
from chemistrylab.extract_bench.extract_bench_v1_engine import ExtractBenchEnv

def get_extract_vessel(vessel_path=None, extract_vessel=None):
    '''
    Function to obtain an extraction vessel containing materials.

    Parameters
    ---------------
    `vessel_path` : `str` (default=`None`)
        A string indicating the local of a pickle file containing the required vessel object.
    `extract_vessel` : `vessel` (default=`None`)
        A vessel object containing state variables, materials, solutes, and spectral data.

    Returns
    ---------------
    `extract_vessel` : `vessel`
        A vessel object containing state variables, materials, solutes, and spectral data.

    Raises
    ---------------
    `IOError`:
        Raised if neither a path to a pickle file nor a vessel object is provided.
    '''

    # ensure that at least one of the methods to obtain a vessel is provided
    if all([not os.path.exists(vessel_path), not extract_vessel]):
        raise IOError("No vessel acquisition method specified.")

    # if a vessel object is provided, use it;
    # otherwise locate and extract the vessel object as a pickle file
    if not extract_vessel:
        with open(vessel_path, 'rb') as open_file:
            extract_vessel = pickle.load(open_file)

    return extract_vessel

def oil_vessel():
    '''
    Function to generate an input vessel for the oil and water extraction experiment.

    Parameters
    ---------------
    None

    Returns
    ---------------
    `extract_vessel` : `vessel`
        A vessel object containing state variables, materials, solutes, and spectral data.

    Raises
    ---------------
    None
    '''

    # initialize extraction vessel
    extraction_vessel = vessel.Vessel(label='extraction_vessel')

    # initialize H2O
    H2O = material.H2O

    # initialize Na
    Na = material.Na
    Na().set_charge(1.0)
    Na().set_solute_flag(True)
    Na().set_polarity(2.0)

    # initialize Cl
    Cl = material.Cl
    Cl().set_charge(-1.0)
    Cl().set_solute_flag(True)
    Cl().set_polarity(2.0)

    # material_dict
    material_dict = {
        H2O().get_name(): [H2O, 30.0, 'mol'],
        Na().get_name(): [Na, 1.0, 'mol'],
        Cl().get_name(): [Cl, 1.0, 'mol']
    }

    # solute_dict
    solute_dict = {
        Na().get_name(): {H2O().get_name(): [1]},
        Cl().get_name(): {H2O().get_name(): [1]},
    }

    material_dict, solute_dict, _ = util.check_overflow(
        material_dict=material_dict,
        solute_dict=solute_dict,
        v_max=extraction_vessel.get_max_volume()
    )

    # set events and push them to the queue
    extraction_vessel.push_event_to_queue(
        events=None,
        feedback=[
            ['update material dict', material_dict],
            ['update solute dict', solute_dict],
            ['fully mix']
        ],
        dt=0
    )

    return extraction_vessel

class ExtractWorld_v1(ExtractBenchEnv):
    '''
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    '''

    def __init__(self):
        super(ExtractWorld_v1, self).__init__(
            extraction='wurtz',
            extraction_vessel=get_extract_vessel(
                vessel_path=os.path.join(os.getcwd(), "react_vessel.pickle"),
                extract_vessel=None
            ),
            solute="H2O",
            target_material='dodecane',
            out_vessel_path=os.getcwd()
        )

class ExtractWorld_v2(ExtractBenchEnv):
    '''
    Class to define an environment which performs a water-oil extraction on materials in a vessel.
    '''

    def __init__(self):
        super(ExtractWorld_v2, self).__init__(
            extraction='water_oil',
            extraction_vessel=oil_vessel(),
            target_material='Na'
        )

