from typing import NamedTuple, Tuple, Callable, Optional, List
import numpy as np
import json

import os
REACTION_PATH = os.path.dirname(__file__)+"/available_reactions"


def serial(x):
    """Turn a numpy array into tuples"""
    try:return float(x)
    except:return tuple(a for a in x)



def format_2d_array_string(array_str, indent=2):
    """
    Special json formatting function to create pretty arrays
    
    Args:
        array_str (str): The json string
        indent (int): The amount of indentation in the json string

    Returns:
        str: A prettier version of the json string
    
    """
    ind=" "*indent
    # Replace all newlines after numbers with spaces
    for i in range(10):
        array_str = array_str.replace(f"{i},\n", f"{i}, ")
        array_str = array_str.replace(f"{i}\n", f"{i}")
    
    # replace newlines after opening brackets with spaces
    array_str = array_str.replace(f"[\n", f"[ ")
    
    #add back in newlines if there was a string after
    array_str = array_str.replace(f': [ {ind*2}"', f': [\n{ind*2}"')
    #same thing if there was another opening bracket after
    array_str = array_str.replace(f': [ {ind*2}[', f': [\n{ind*2}[')
    # adjust spacing for negative signs
    array_str = array_str.replace(f'{ind*3} -', '-')
    
    #adjust spacing of array numbers
    for i in range(10):
        array_str = array_str.replace(f" {ind*3}{i}", f" {i}")
        array_str = array_str.replace(f"{i}{ind*2}", f"{i} ")
        array_str = array_str.replace(f"{ind*2}{i}", f"{i}")
    return array_str


class ReactInfo(NamedTuple):
    name: str
    REACTANTS:        Tuple[str]
    PRODUCTS:         Tuple[str]
    SOLVENTS:         Tuple[str]
    MATERIALS:        Tuple[str]
    pre_exp_arr:      np.array
    activ_energy_arr: np.array
    stoich_coeff_arr: np.array
    conc_coeff_arr:   np.array
        
    def dump_to_json(self,fn):
        """
        Saves the reaction information to a json file
        
        Args:
            fn (str): The filename to save as.
        """
        with open(fn,"w") as f:
            f.write(
            format_2d_array_string(
                json.dumps(self._asdict(), indent=2, default=serial),
                indent=2
                )
            )
    @staticmethod
    def from_json(fn):
        """
        Creates a new ReactInfo object from a json file
        Args:
            fn (str): The filename to load from

        Returns:
            ReactInfo: The ReactInfo object parameterized by the json file
        
        """
        with open(fn,"r") as f:
            kwargs = json.load(f)
        modified_kwargs = {i: np.array(kwargs[i]) if "_arr" in i else kwargs[i] for i in kwargs}
        return ReactInfo(**modified_kwargs)

ReactInfo.name.__doc__ = "The name of the reaction"
ReactInfo.REACTANTS.__doc__ = "The reactants involved in the reaction(s)"
ReactInfo.PRODUCTS.__doc__ = "Any products formed in the reaction(s)"
ReactInfo.SOLVENTS.__doc__ = "Any solvents used in the reaction(s)"
ReactInfo.MATERIALS.__doc__ = "All materials or substances involved in the reaction(s) (a union of reactants products and solvents)"
ReactInfo.pre_exp_arr.__doc__ = "The pre-exponential factors for the reaction(s)"
ReactInfo.activ_energy_arr.__doc__ = "The activation energies for the reaction(s)"
ReactInfo.stoich_coeff_arr.__doc__ = "Stoichiometric coefficients of the reactants in each reaction"
ReactInfo.conc_coeff_arr.__doc__ = "The concentration coefficients of all involved materials in each reaction"