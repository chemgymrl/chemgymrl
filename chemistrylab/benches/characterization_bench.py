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
"""

import numpy as np
import copy
import sys
sys.path.append("../../")

import numba

@numba.jit(nopython=True)
def calc_absorb3(item, C, x, w_min, w_max, absorb):
     
    # iterate through the spectral parameters in self.params and the wavelength space
    for j in range(item.shape[0]):
        nm = abs((1e7/item[j,1]-w_min)/(w_min-w_max))
        #get a bounding area for x values above 1e-12
        radius = 6*item[j, 2]*x.shape[0]
        left = max(0,int(nm*x.shape[0]-radius-1))
        right = min(x.shape[0],int(nm*x.shape[0]+radius+2))
        for k in range(left,right):
            #convert inverse cm to nm
            
            decay_rate = np.exp(
                -0.5 * (
                    (x[k] - nm) / item[j, 2]
                ) ** 2.0
            )
            if decay_rate < 1e-30:
                decay_rate = 0
            absorb[k] += C * item[j, 0] * decay_rate


class CharacterizationBench:
    """
    A set of methods made available to inspect an inputted vessel.

    Args:
    - observation_list (Tuple[str]): Ordered list of observations to make (see Method Map)
    - targets (Tuple[str]): A list of target materials
    - n_vessels (int): The (maximum) number of vessels included in an observation

    Method Map:
    - 'spectra' -> get_spectra
    - 'layers'  -> get_layers
    - 'targets' -> encode_target
    - 'PVT'     -> encode_PVT

    """

    def __init__(self, observation_list,targets,n_vessels):

        # specify the analysis techniques available in this bench
        self.techniques = {'spectra': self.get_spectra}
        self.params = {'spectra': {'range_ir': (2000, 20000)}}

        self.targets=targets
        self.n_vessels=n_vessels
        self.sizes=dict(
            spectra=200,
            layers=100,
            targets=len(targets),
            PVT = 3
        )

        #List of all functions, this can probably be a class variable instead of an instance variable
        self.all_functions = dict(
            spectra=self.get_spectra,
            layers=self.get_layers,
            targets=self.encode_target,
            PVT=self.encode_PVT
        )

        #create a list of functions which will be used to build your observations
        self.observation_list=observation_list
        self.functions = [self.all_functions[a] for a in observation_list]
        
        #Determine the shape of the observations
        n_pixels=sum(self.sizes[a] for a in observation_list)
        self.observation_shape=(n_vessels,n_pixels)

    def get_observation(self, vessels, target):
        """
        Returns a concatenation of observations of the vessels provided, using the list of observations provided
        in __init__

        Args:
        - vessels (Tuple[Vessel]): A list of vessels you want an observtation of.
        - target (str): The current target material
        """
        self.target=target
        state=np.zeros(self.observation_shape)
        for i,v in enumerate(vessels):
            if i>= self.n_vessels:break
            state[i] = np.concatenate([f(v) for f in self.functions])
        return state

    def __call__(self, vessels, target):
        return self.get_observation(vessels, target)


    def get_spectra(self, vessel, materials=None, overlap=True):
        """
        Class method to generate total spectral data using a gaussian decay.

        Args:
        - vessel (Vessel): The vessel inputted for spectroscopic analysis.
        - materials (Optional[list]): List of materials to get the spectra from
        - overlap (bool): Indicates if the spectral plots show overlapping signatures. 
                Defaults to False.

        Returns:
        - spectra (np.array): A 1D array containing the absorption data of the present materials.

        Note: Note sure if this is improved. . . consider falling back to previous implementation: calc_absorb
        """
        
        mat_dict=vessel.material_dict
        # acquire the array of material concentrations
        if not materials:
            materials = [mat for key,mat in mat_dict.items()]
        else:
            materials = [mat_dict[mat] for mat in materials if mat in mat_dict]

        #prepare data to store absorption
        w_min,w_max=self.params['spectra']['range_ir']
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
        absorb = np.zeros(x.shape[0], dtype=np.float32)

        #Get concentrations and spectras
        C = np.array([mat.mol for mat in materials])/vessel.filled_volume()
        if not overlap:
            params = tuple(mat.get_spectra_no_overlap() for mat in materials)
        else:
            params = tuple(mat.get_spectra_overlap() for mat in materials)

        #Build observation
        for i,param in enumerate(params):
            calc_absorb3(param, C[i], x, w_min, w_max, absorb)
        return np.clip(absorb, 0.0, 1.0)

            
    def get_layers(self, vessel):
        """Returns a 1D array of vessel layer information"""
        return vessel.get_layers()

    def encode_PVT(self,vessel):
        """Returns a size 3 array containing [temperature,volume,pressure]"""
        # set up the temperature
        #Tmin, Tmax = vessel.get_Tmin(), vessel.get_Tmax()
        temp = vessel.temperature
        normalized_temp = (temp - 0) / (1000 - 0)
        # set up the volume: this measure is kind of bad
        #consider using (vessel.get_current_volume()[-1] / vessel.get_volume()) instead
        #Vmin, Vmax = vessel.get_min_volume(), vessel.get_max_volume()
        volume = vessel.filled_volume()
        normalized_volume = (volume - 0) / (vessel.volume - 0)
        # set up the pressure
        #Pmax = vessel.get_pmax()
        #total_pressure = vessel.get_pressure()
        normalized_pressure = 0#total_pressure / Pmax
        #add them all into a 1D array
        out = np.array([normalized_temp,normalized_volume,normalized_pressure],dtype=np.float32)
        return np.clip(out,0,1)

    def encode_target(self, vessel):
        """
        Returns a 1D one-hot encoding of the target material (note self.target must be set before doing this)
        """
        targ_index = self.targets.index(self.target)
        one_hot=np.zeros(len(self.targets))
        one_hot[targ_index]=1

        return one_hot