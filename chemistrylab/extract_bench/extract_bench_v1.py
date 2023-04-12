"""
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

Module to access and execute the ExtractWorld Engine

:title: extractworld_v1.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-06-24
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

# import external modules
import os
import pickle
import sys
from random import choice
from copy import deepcopy

# import local modules
sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.chem_algorithms import material, util, vessel
from chemistrylab.extract_bench.extract_bench_v1_engine import ExtractBenchEnv
from chemistrylab.reactions.reaction_base import _Reaction

from chemistrylab.extract_algorithms.extractions.extraction_2 import Extraction as Extraction2

def get_extract_vessel(vessel_path=None, extract_vessel=None):
    """
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
    """

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
    extraction_vessel = vessel.Vessel(label='extraction_vessel')

    # initialize H2O
    H2O = material.H2O()

    # initialize Na
    Na = material.Na()
    Na.set_charge(1.0)
    Na.set_solute_flag(True)
    Na.set_polarity(2.0)
    Na.set_phase('l')
    Na._boiling_point = material.NaCl()._boiling_point

    # initialize Cl
    Cl = material.Cl()
    Cl.set_charge(-1.0)
    Cl.set_solute_flag(True)
    Cl.set_polarity(2.0)
    Cl.set_phase('l')
    Cl._boiling_point = material.NaCl()._boiling_point

    # material_dict
    material_dict = {
        H2O.get_name(): [H2O, 30.0, 'mol'],
        Na.get_name(): [Na, 1.0, 'mol'],
        Cl.get_name(): [Cl, 1.0, 'mol']
    }

    # solute_dict
    solute_dict = {
        Na.get_name(): {H2O.get_name(): [H2O, 1, 'mol']},
        Cl.get_name(): {H2O.get_name(): [H2O, 1, 'mol']},
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

def wurtz_vessel(add_mat=""):
    """
    Function to generate an input vessel for the wurtz extraction experiment.
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
    extraction_vessel = vessel.Vessel(label='extraction_vessel')

    # initialize DiEthylEther
    DiEthylEther = material.DiEthylEther()

    # initialize Na
    Na = material.Na()
    Na.set_charge(1.0)
    Na.set_solute_flag(True)
    Na.set_color(0.0)
    Na.set_polarity(2.0)
    Na.set_phase('l')
    Na._boiling_point = material.NaCl()._boiling_point

    # initialize Cl
    Cl = material.Cl()
    Cl.set_charge(-1.0)
    Cl.set_solute_flag(True)
    Cl.set_color(0.0)
    Cl.set_polarity(2.0)
    Cl.set_phase('l')
    Cl._boiling_point = material.NaCl()._boiling_point

    # initialize material
    products = {'dodecane': material.Dodecane,
        '5-methylundecane': material.FiveMethylundecane,
        '4-ethyldecane': material.FourEthyldecane,
        '5,6-dimethyldecane': material.FiveSixDimethyldecane,
        '4-ethyl-5-methylnonane': material.FourEthylFiveMethylnonane,
        '4,5-diethyloctane': material.FourFiveDiethyloctane,
        'NaCl': material.NaCl
    }
    try:
        if add_mat == "":
            add_mat = choice(list(products.keys()))
        
        if add_mat != "NaCl":
            add_material = products[add_mat]()
        else:
            add_material = products['dodecane']()
    
    except KeyError:
        add_mat = 'dodecane'
        add_material = products['dodecane']()
    
    add_material.set_solute_flag(True)
    add_material.set_color(0.0)
    add_material.set_phase('l')

    # material_dict
    material_dict = {
        DiEthylEther.get_name(): [DiEthylEther, 4.0, 'mol'],
        Na.get_name(): [Na, 1.0, 'mol'],
        Cl.get_name(): [Cl, 1.0, 'mol'],
        add_material.get_name(): [add_material, 1.0, 'mol']
    }

    # solute_dict
    solute_dict = {
        Na.get_name(): {DiEthylEther.get_name(): [DiEthylEther, 1.0, 'mol']},
        Cl.get_name(): {DiEthylEther.get_name(): [DiEthylEther, 1.0, 'mol']},
        add_material.get_name(): {DiEthylEther.get_name(): [DiEthylEther, 1.0, 'mol']}
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

    return extraction_vessel, add_mat

class WurtzExtract_v1(ExtractBenchEnv):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self):
        super(WurtzExtract_v1, self).__init__(
            extraction_vessel=wurtz_vessel('dodecane')[0],
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz",
            n_steps=50,
            target_material='dodecane',
            solvents=["C6H14", "diethyl ether"],
            out_vessel_path=None
        )
        
class WurtzExtract_v2(ExtractBenchEnv):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self):
        super(WurtzExtract_v2, self).__init__(
            extraction_vessel=wurtz_vessel('dodecane')[0],
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz",
            n_steps=50,
            target_material='dodecane',
            solvents=["C6H14", "diethyl ether"],
            out_vessel_path=None,
            Extraction=Extraction2
        )

class GeneralWurtzExtract_v1(ExtractBenchEnv):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self, target_material="", in_vessel_path=None):
        self.original_target_material = target_material
        extract_vessel, target_mat = wurtz_vessel(self.original_target_material)

        super(GeneralWurtzExtract_v1, self).__init__(
            extraction_vessel=extract_vessel,
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz",
            in_vessel_path=in_vessel_path,
            n_steps=50,
            solvents=["C6H14", "diethyl ether"],
            target_material=target_mat,
            out_vessel_path=None
        )

    def reset(self):
        extract_vessel, target_mat = wurtz_vessel(self.original_target_material)
        self.original_extraction_vessel = deepcopy(extract_vessel)
        self.target_material = target_mat

        return super(GeneralWurtzExtract_v1, self).reset()
    
    
class GeneralWurtzExtract_v2(ExtractBenchEnv):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self, target_material="", in_vessel_path=None):
        self.original_target_material = target_material
        extract_vessel, target_mat = wurtz_vessel(self.original_target_material)

        super(GeneralWurtzExtract_v2, self).__init__(
            extraction_vessel=extract_vessel,
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz",
            in_vessel_path=in_vessel_path,
            n_steps=50,
            solvents=["C6H14", "diethyl ether"],
            target_material=target_mat,
            out_vessel_path=None,
            Extraction=Extraction2
        )

    def reset(self):
        extract_vessel, target_mat = wurtz_vessel(self.original_target_material)
        self.original_extraction_vessel = deepcopy(extract_vessel)
        self.target_material = target_mat

        return super(GeneralWurtzExtract_v2, self).reset()


class WurtzExtractCtd_v1(ExtractBenchEnv):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self):
        super(WurtzExtractCtd_v1, self).__init__(
            extraction_vessel=get_extract_vessel(
                vessel_path=os.path.join(os.getcwd(), "react_vessel.pickle"),
                extract_vessel=vessel.Vessel(label='temp')
            ),
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz",
            n_steps=50,
            target_material='dodecane',
            solvents=["C6H14", "diethyl ether"],
            out_vessel_path=None
        )

class WaterOilExtract_v1(ExtractBenchEnv):
    """
    Class to define an environment which performs a water-oil extraction on materials in a vessel.
    """

    def __init__(self):
        super(WaterOilExtract_v1, self).__init__(
            reaction=_Reaction,
            reaction_file_identifier="decomp",
            solvents=["C6H14", "H2O"],
            extraction_vessel=oil_vessel(),
            n_steps=50,
            target_material='Na',
            out_vessel_path=None
        )
