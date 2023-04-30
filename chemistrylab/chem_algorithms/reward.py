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

Module to define a universal reward function.
'''

import copy
import math
import numpy as np
import sys

sys.path.append("../../")
from chemistrylab.chem_algorithms import material


def get_dissolved_amounts(vessel, desired_material):
    """    
    Returns:
    - min_amount (float): The amount material that could be produced if you removed the solvent
        This is the minimum of (quantity/stoich_coeff) for each dissolved component
    - contributions (float): Amount of mols of solutes to subtract from the total material amount
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
    def __init__(self, use_purity, exclude_solvents, include_dissolved, exclude_mat=None):
        self.exclude_solvents=exclude_solvents
        self.include_dissolved=include_dissolved
        self.use_purity=use_purity
        self.exclude_mat=exclude_mat
    def __call__(self,vessels,desired_material,exclude_material = None):
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
            
