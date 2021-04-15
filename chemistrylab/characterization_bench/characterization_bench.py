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

Characterization Bench Class

:title: characterization_bench

:author: Nicholas Paquin

:history: 26-03-2021

Class to initialize reaction vessels, prepare the reaction base environment, and designate actions to occur.
"""

import numpy as np


class CharacterizationBench:
    """
    Class to define the characterization bench and its methods made available to inspect an inputted vessel.
    """

    def __init__(self):
        """
        Constructor class method for the analysis bench.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # specify the analysis techniques available in this bench
        self.techniques = {'spectra': self.get_spectra}

    def analyze(self, vessel, analysis, overlap=False):
        """
        Constructor class method to pass thermodynamic variables to class methods.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # perform the specified analysis technique
        analysis = self.techniques[analysis](vessel, overlap)
        return analysis, vessel

    @staticmethod
    def get_spectra(vessel, materials=None, overlap=True):
        """
        Class method to generate total spectral data using a guassian decay.

        Parameters
        ---------------
        `vessel` : `vessel.Vessel`
            The vessel inputted for spectroscopic analysis.
        `overlap` : `boolean` (default=`False`)
            Indicates if the spectral plots show overlapping signatures.

        Returns
        ---------------
        `absorb` : `np.array`
            An array of the total absorption data of every chemical in the experiment

        Raises
        ---------------
        None
        """

        # acquire the array of material concentrations
        if not materials:
            materials = list(vessel.get_material_dict().keys())

        C = vessel.get_concentration(materials=materials)
        mat_dict = vessel.get_material_dict()

        if not overlap:
            params = [mat_dict[mat][0]().get_spectra_no_overlap() if mat in mat_dict else None for mat in materials]
        else:
            params = [mat_dict[mat][0]().get_spectra_overlap() if mat in mat_dict else None for mat in materials]

        # set the wavelength space
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)

        # define an array to contain absorption data
        absorb = np.zeros(x.shape[0], dtype=np.float32)

        # iterate through the spectral parameters in self.params and the wavelength space
        for i, item in enumerate(params):
            if item is not None:
                for j in range(item.shape[0]):
                    for k in range(x.shape[0]):
                        amount = C[i]/sum(C)
                        height = item[j, 0]
                        decay_rate = np.exp(
                            -0.5 * (
                                (x[k] - params[i][j, 1]) / params[i][j, 2]
                            ) ** 2.0
                        )
                        if decay_rate < 1e-30:
                            decay_rate = 0
                        absorb[k] += amount * height * decay_rate

        # absorption must be between 0 and 1
        absorb = np.clip(absorb, 0.0, 1.0)

        return absorb

    @staticmethod
    def get_spectra_peak(vessel, materials=None, overlap=True):
        """
        Method to populate a list with the spectral peak of each chemical.

        Parameters
        ---------------
        `C` : `np.array`
            An array containing the concentrations of all materials in the vessel.
        `params` : `list`
            A list of spectra overlap parameters.
        `materials` : `list`
            A list of the materials in the inputted vessel.

        Returns
        ---------------
        `spectra_peak` : `list`
            A list of parameters specifying the peak of the spectra for each chemical.

        Raises
        ---------------
        None
        """
        if not materials:
            materials = list(vessel.get_material_dict().keys())

        C = vessel.get_concentration(materials=materials)
        mat_dict = vessel.get_material_dict()

        if not overlap:
            params = [mat_dict[mat][0]().get_spectra_no_overlap() if mat in mat_dict else None for mat in materials]
        else:
            params = [mat_dict[mat][0]().get_spectra_overlap() if mat in mat_dict else None for mat in materials]

        # create a list of the spectral peak of each chemical
        spectra_peak = []
        for i, material in enumerate(materials):
            if params[i] is not None:
                spectra_peak.append([
                    params[i][:, 1] * 600 + 200,
                    C[i] * params[i][:, 0],
                    material
                ])

        return spectra_peak

    @staticmethod
    def get_dash_line_spectra(vessel, materials=None, overlap=True):
        """
        Module to generate each individual spectral dataset using gaussian decay.

        Parameters
        ---------------
        `C` : `np.array`
            An array containing the concentrations of all materials in the vessel.
        `params` : `list`
            A list of spectra overlap parameters.

        Returns
        ---------------
        dash_spectra : list
            A list of all the spectral data of each chemical

        Raises
        ---------------
        None
        """
        if not materials:
            materials = list(vessel.get_material_dict().keys())

        C = vessel.get_concentration(materials=materials)
        mat_dict = vessel.get_material_dict()

        if not overlap:
            params = [mat_dict[mat][0]().get_spectra_no_overlap() if mat in mat_dict else None for mat in materials]
        else:
            params = [mat_dict[mat][0]().get_spectra_overlap() if mat in mat_dict else None for mat in materials]

        dash_spectra = []

        # set the wavelength space
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)

        for i, item in enumerate(params):
            if item is not None:
                each_absorb = np.zeros(x.shape[0], dtype=np.float32)
                for j in range(item.shape[0]):
                    for k in range(x.shape[0]):
                        amount = C[i]
                        height = item[j, 0]
                        decay_rate = np.exp(
                            -0.5 * (
                                (x[k] - params[i][j, 1]) / params[i][j, 2]
                            ) ** 2.0
                        )
                        each_absorb += amount * height * decay_rate
                dash_spectra.append(each_absorb)

        return dash_spectra
