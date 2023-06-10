'''
This module provides a default set of reward functions.
'''

import copy
import math
import numpy as np
import sys

from chemistrylab import material
from chemistrylab.vessel import Vessel
from typing import NamedTuple, Tuple, Callable, Optional, List



def get_dissolved_amounts(vessel: Vessel, desired_material: str):
    """    
    Returns:
        Tuple[float,float]: 
            - The amount material that could be produced if you removed the solvent. This is the minimum of (quantity/stoich_coeff) for each dissolved component
            - The amount of mols of solutes to subtract from the total material amount
    """
    material_amounts = []
    dis_mats = material.REGISTRY[desired_material]().dissolve()
    # Dissolved version is already the target material
    if len(dis_mats) < 2: return 0,0
    min_amount=float("inf")
    n=0
    for mat in dis_mats:
        #Can't make any with what's dissolved
        if not mat._name in vessel.solute_dict: return 0,0
        # Determine how much of the target the solute would make
        amount = vessel.material_dict[mat._name].mol / dis_mats[mat]
        if amount< min_amount:
            min_amount=amount
        n+=dis_mats[mat]

    contributions=min_amount*(n-1)
    return min_amount,contributions
        
class RewardGenerator():
    """
    RewardGenerator class generates rewards for a given set of vessels and desired materials.

    Args:
        use_purity (bool): True if reward is based on purity, False if reward is based on the amount of desired material.
        exclude_solvents (bool): True if solvents should be excluded from the total amount of materials for purity calculations.
        include_dissolved (bool): True if reward should include dissolved material components in the vessels as the desired material.
        exclude_mat (str, optional): A string representing a material which gives a negative reward.
    
    This class returns callable objects, which serve as reward functions
    """
    def __init__(self, use_purity, exclude_solvents, include_dissolved, exclude_mat=None):
        self.exclude_solvents=exclude_solvents
        self.include_dissolved=include_dissolved
        self.use_purity=use_purity
        self.exclude_mat=exclude_mat
    def __call__(self,vessels: Tuple[Vessel], desired_material: str, exclude_material: Optional[str] = None):
        """
        Assign a reward to a set of vessels based off of what is desired/undesired

        Args:
            vessels (Tuple[Vessel]): A list of Vessel objects.
            desired_material (str): A string representing the desired material for which the reward should be calculated.
            exclude_material (Optional[str]): Currently unused
    
        Returns:
            float: A floating point number representing the calculated reward.

        """
        reward=0
        for v in vessels:
            mat_dict = v.material_dict
            # Get the amount of target material that could be extracted from dissolved components
            if self.include_dissolved:
                dissolved, exclude= get_dissolved_amounts(v,desired_material)
            else:
                dissolved=exclude=0
            # Grab the amount of target material
            amount=dissolved+(mat_dict[desired_material].mol if desired_material in mat_dict else 0)
            if self.exclude_mat is not None and (self.exclude_mat != desired_material):
                amount -= (mat_dict[self.exclude_mat].mol if self.exclude_mat in mat_dict else 0)
            #Purity requires you to multiply by (desired_amount)/(total_amount)
            if self.use_purity:
                #Total amount is either calculated using all materials, or just the non-solvents
                if self.exclude_solvents:
                    all_mats = sum(mat.mol for key,mat in mat_dict.items() if not mat.is_solvent())
                else:
                    all_mats = sum(mat.mol for key,mat in mat_dict.items())
                all_mats -= exclude
                if all_mats>0:
                    reward += amount**2/all_mats
            else:
                reward+= amount
        return reward
            
