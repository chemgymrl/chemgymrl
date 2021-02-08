'''
Distillation Bench Environment

:title: distillation_bench_v1.py

:author: Mitchell Shahen

:history: 2020-07-23
'''

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

# import external modules
import os
import pickle
import sys

# import local modules
sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.distillation_bench.distillation_bench_v1_engine import DistillationBenchEnv

def get_vessel(vessel_path=None, in_vessel=None):
    '''
    Function to obtain a vessel.

    Parameters
    ---------------
    `vessel_path` : `str` (default=`None`)
        A string indicating the local of a pickle file containing the required vessel object.
    `in_vessel` : `vessel` (default=`None`)
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
    if all([not os.path.exists(vessel_path), not in_vessel]):
        raise IOError("Invalid vessel acquisition method specified.")

    # if a vessel object is provided, use it;
    # otherwise locate and extract the vessel object as a pickle file
    if not in_vessel:
        with open(vessel_path, 'rb') as open_file:
            in_vessel = pickle.load(open_file)

    return in_vessel

class Distillation_v1(DistillationBenchEnv):
    '''
    Class to define an environment to perform a distillation experiment
    on an inputted vessel and obtain a pure form of a targetted material.
    '''

    def __init__(self):
        super(Distillation_v1, self).__init__(
            boil_vessel=get_vessel(
                vessel_path=os.path.join(os.getcwd(), "test_extract_vessel.pickle"),
                in_vessel=None
            ),
            target_material="dodecane",
            out_vessel_path=os.getcwd()
        )
