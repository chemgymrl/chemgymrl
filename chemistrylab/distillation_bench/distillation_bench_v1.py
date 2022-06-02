'''
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.

Distillation Bench Environment
:title: distillation_bench_v1.py
:author: Mitchell Shahen, Mark Baula
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
from chemistrylab.chem_algorithms import material, util, vessel
from chemistrylab.reactions.reaction_base import _Reaction

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
                in_vessel=boil_vessel()
            ),
            target_material="dodecane",
            dQ=1000.0,
            out_vessel_path=os.getcwd()
        )

def wurtz_vessel(add_mat):
    """
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
    """

    # initialize extraction vessel
    extraction_vessel = vessel.Vessel(label='boil_vessel')

    # initialize C6H14
    C6H14 = material.C6H14()

    # initialize material
    products = {'dodecane': material.Dodecane,
        '5-methylundecane': material.FiveMethylundecane,
        '4-ethyldecane': material.FourEthyldecane,
        '5,6-dimethyldecane': material.FiveSixDimethyldecane,
        '4-ethyl-5-methylnonane': material.FourEthylFiveMethylnonane,
        '4,5-diethyloctane': material.FourFiveDiethyloctane
    }
    try:
        add_material = products[add_mat]()
    except KeyError:
        add_material = products['dodecane']()
    add_material.set_solute_flag(True)
    add_material.set_color(0.0)
    add_material.set_phase('l')

    # material_dict
    material_dict = {
        C6H14.get_name(): [C6H14, 4.0, 'mol'],
        add_material.get_name(): [add_material, 1.0, 'mol']
    }

    # solute_dict
    solute_dict = {
        add_material.get_name(): {C6H14.get_name(): [C6H14, 1.0, 'mol']}
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
            ['update solute dict', solute_dict]
        ],
        dt=0
    )
    extraction_vessel.push_event_to_queue(
        events=None,
        feedback=None,
        dt=-100000
    )

    return extraction_vessel

class WurtzDistill_v1(DistillationBenchEnv):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self):
        super(WurtzDistill_v1, self).__init__(
            boil_vessel=wurtz_vessel(),
            n_vessel_pixels=100,
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz",
            in_vessel_path=None,
            target_material="dodecane",
            dQ=1000.0,
            out_vessel_path=os.getcwd()
        )

class GeneralWurtzDistill_v1(DistillationBenchEnv):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self, target_material, in_vessel_path=None):
        super(GeneralWurtzDistill_v1, self).__init__(
            boil_vessel=wurtz_vessel(target_material),
            n_vessel_pixels=100,
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz",
            in_vessel_path=in_vessel_path,
            target_material=target_material,
            dQ=1000.0,
            out_vessel_path=os.getcwd()
        )

def boil_vessel():
    """
    Function to generate an input vessel for distillation.

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
    """

    # initialize boiling vessel
    boil_vessel = vessel.Vessel(label='boil_vessel')

    # initialize materials
    OneChlorohexane = material.OneChlorohexane
    TwoChlorohexane = material.TwoChlorohexane
    ThreeChlorohexane = material.ThreeChlorohexane
    Na = material.Na
    Dodecane = material.Dodecane
    FiveMethylundecane = material.FiveMethylundecane
    FourEthyldecane = material.FourEthyldecane
    FiveSixDimethyldecane = material.FiveSixDimethyldecane
    FourEthylFiveMethylnonane = material.FourEthylFiveMethylnonane
    FourFiveDiethyloctane = material.FourFiveDiethyloctane
    NaCl = material.NaCl
    H2O = material.H2O

    # material_dict
    material_dict = {
        OneChlorohexane().get_name(): [OneChlorohexane, 0.62100387, 'mol'],
        TwoChlorohexane().get_name(): [TwoChlorohexane, 0.71239483, 'mol'],
        ThreeChlorohexane().get_name(): [ThreeChlorohexane, 0.6086047, 'mol'],
        Na().get_name(): [Na, 0.028975502, 'mol'],
        Dodecane().get_name(): [Dodecane, 0.10860507, 'mol'],
        FiveMethylundecane().get_name(): [FiveMethylundecane, 0.07481375, 'mol'],
        FourEthyldecane().get_name(): [FourEthyldecane, 0.08697271, 'mol'],
        FiveSixDimethyldecane().get_name(): [FiveSixDimethyldecane, 0.07173399, 'mol'],
        FourEthylFiveMethylnonane().get_name(): [FourEthylFiveMethylnonane, 0.069323435, 'mol'],
        FourFiveDiethyloctane().get_name(): [FourFiveDiethyloctane, 0.07406321, 'mol'],
        NaCl().get_name(): [NaCl, 0.9710248, 'mol'],
        H2O().get_name(): [H2O, 0.2767138495698029, 'mol']
    }

    # solute_dict
    solute_dict = {
    }

    material_dict, solute_dict, _ = util.check_overflow(
        material_dict=material_dict,
        solute_dict=solute_dict,
        v_max=boil_vessel.get_max_volume()
    )

    # set events and push them to the queue
    boil_vessel.push_event_to_queue(
        events=None,
        feedback=[
            ['update material dict', material_dict],
            ['update solute dict', solute_dict]
        ],
        dt=0
    )

    return boil_vessel

