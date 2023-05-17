from typing import NamedTuple, Tuple, Callable, Optional, List
import numpy as np
import json

import os
REACTION_PATH = os.path.dirname(__file__)+"/available_reactions"


def serial(x):
    """Turn a 1D or 2D array into a tuple"""
    try:
        return tuple(tuple(a) for a in x)
    except:
        return tuple(a for a in x)



def format_2d_array_string(array_str,indent=2):
    """Messes around with json spacing for arrays to get a nicer output"""
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
        """Saves the reaction information to a json file"""
        with open(fn,"w") as f:
            f.write(
            format_2d_array_string(
                json.dumps(self._asdict(), indent=2, default=serial),
                indent=2
                )
            )
    @staticmethod
    def from_json(fn):
        """Creates a new ReactInfo object from a json file"""
        with open(fn,"r") as f:
            kwargs = json.load(f)
        modified_kwargs = {i: np.array(kwargs[i]) if "_arr" in i else kwargs[i] for i in kwargs}
        return ReactInfo(**modified_kwargs)