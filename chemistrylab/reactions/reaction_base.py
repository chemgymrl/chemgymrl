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

Module to perform a series of reactions as specified by an accompanying reaction file.

:title: reaction_base.py

:author: Mark Baula, Nicholas Paquin, and Mitchell Shahen

:history: 22-06-2020
"""

# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=unsubscriptable-object

import importlib.util
import os
import sys
import types

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../")  # allows the module to access chemistrylab

from chemistrylab.reactions.get_reactions import convert_to_class
from chemistrylab.characterization_bench.characterization_bench import CharacterizationBench
from chemistrylab.chem_algorithms import vessel
from chemistrylab.lab.de import De
from scipy.integrate import solve_ivp

R = 8.314462619


class _Reaction:

    def __init__(self, reaction_file_identifier="", overlap=False, solver='RK45'):
        """
        Constructor class module for the Reaction class.

        Parameters
        ---------------
        `reaction_file_identifier` : `str` (default=`""`)
            The basename of the reaction file that contains the parameters
            necessary to perform the desired reaction(s).
        `overlap` : `bool` (default=`False`)
            Indicate if the spectral plot includes overlapping plots.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # define a name/label for this reaction base class
        self.name = "base_reaction"

        # ensure the requested reaction file is found and accessible
        reaction_filepath = self._find_reaction_file(reaction_file=reaction_file_identifier)

        # acquire the necessary parameters from the reaction file
        reaction_params = self._get_reaction_params(reaction_filepath=reaction_filepath)

        # UNPACK THE REACTION PARAMETERS:
        # materials used and the desired material
        self.reactants = reaction_params["REACTANTS"]
        self.products = reaction_params["PRODUCTS"]
        self.solutes = reaction_params["SOLUTES"]
        self.desired = reaction_params["DESIRED"]

        # initial vessel properties (used in reseting the reaction vessel)
        self.Ti = reaction_params["Ti"]
        self.Vi = reaction_params["Vi"]

        # additional vessel properties (to be assigned during the reset function)
        self.Tmin = reaction_params["Tmin"]
        self.Tmax = reaction_params["Tmax"]
        self.Vmin = reaction_params["Vmin"]
        self.Vmax = reaction_params["Vmax"]

        # thermodynamic increment values (used when performing the reactions)
        self.dt = reaction_params["dt"]
        self.dT = reaction_params["dT"]
        self.dV = reaction_params["dV"]

        # the arrays used in dictating how reaction calculations are performed
        self.activ_energy_arr = reaction_params["activ_energy_arr"]
        self.stoich_coeff_arr = reaction_params["stoich_coeff_arr"]
        self.conc_coeff_arr = reaction_params["conc_coeff_arr"]

        self.de = De(self.stoich_coeff_arr, self.activ_energy_arr, self.conc_coeff_arr, len(self.reactants))

        # specify the full list of materials
        self.materials = []

        for mat in self.reactants + self.products:
            if mat not in self.materials:
                self.materials.append(mat)

        # set a threshold value for the minimum, non-negligible material molar amount
        self.threshold = 1e-8

        # define these parameters initially as they are to be given properly in the reset function
        self.initial_in_hand = np.zeros(len(self.reactants))
        self.cur_in_hand = np.zeros(len(self.reactants))
        self.initial_solutes = np.zeros(len(self.solutes))

        # convert the reactants and products to their class object representations
        self.reactant_classes = convert_to_class(materials=self.reactants)
        self.product_classes = convert_to_class(materials=self.products)
        self.material_classes = convert_to_class(materials=self.materials)
        self.solute_classes = convert_to_class(materials=self.solutes)

        # create the (empty) n array which will contain the molar amounts of materials in use
        self.n = np.zeros(len(self.materials))

        # define the maximal number of moles available for any chemical
        self.max_mol = 2.0

        # define the maximal number of moles of each chemical allowed at one time
        self.nmax = self.max_mol * np.ones(len(self.materials))

        # define variables to contain the initial materials and solutes available "in hand"
        self.initial_in_hand = np.zeros(len(self.reactants))
        self.initial_solutes = np.zeros(len(self.solutes))

        # include the available solvers
        self.solvers = {'RK45', 'RK23', 'DOP853', 'DBF', 'LSODA'}

        # select the intended solver or use the default
        if solver in self.solvers:
            self.solver = solver
        else:
            self.solver = 'newton'

        self._solver = solve_ivp

        # define parameters for generating spectra
        self.params = []
        for material in self.material_classes:
            if overlap:
                self.params.append(material().get_spectra_overlap())
            else:
                self.params.append(material().get_spectra_no_overlap())

    @staticmethod
    def _find_reaction_file(reaction_file=""):
        """
        Method to attempt to find the requested reaction file. Currently, the name of the
        reaction file is needed for specifying the requested reaction file.

        Parameters
        ---------------
        `reaction_file` : `str` (default=`""`)
            The basename of the reaction file that contains the parameters
            necessary to perform the desired reaction(s).

        Returns
        ---------------
        `output_reaction_file` : `str`
            The full filepath to the reaction file that contains the parameters
            necessary to perform the desired reaction(s).

        Raises
        ---------------
        `IOError`:
            Raised if the requested reaction file is not found in the available reactions directory.
        """

        # acquire the path to the reactions directory containing
        # this file and the available reactions directory
        reaction_base_filepath = os.path.abspath(__file__)
        reactions_dir = os.path.dirname(reaction_base_filepath)

        # acquire a list of the names of all the available reaction files;
        # specify the directory containing all reaction files
        r_files_dir = os.path.join(reactions_dir, "available_reactions")

        # obtain the list of only files from the above directory
        reaction_files = [
            r_file for r_file in os.listdir(r_files_dir) if os.path.isfile(
                os.path.join(r_files_dir, r_file)
            )
        ]

        # clip the files to omit their extensions (all available reaction files are `.py` files)
        reaction_files_no_ext = [r_file.split(".")[0] for r_file in reaction_files]

        # look for the requested reaction file in the list of available reaction files
        if reaction_file in reaction_files_no_ext:
            output_reaction_file = os.path.join(
                r_files_dir,
                reaction_file + ".py"  # re-add the extension
            )
        else:
            raise IOError("ERROR: Requested Reaction File Not Found in Directory!")

        return output_reaction_file

    @staticmethod
    def _get_reaction_params(reaction_filepath=""):
        """
        Method to acquire the necessary reaction parameters from the specified
        reaction file and make them available.

        Parameters
        ---------------
        `reaction_filepath` : `str` (default=`""`)
            The full filepath to the reaction file that contains the parameters
            necessary to perform the desired reaction(s).

        Returns
        ---------------
        `output_params` : `dict`
            A dictionary containing all of the parameters from the specified reaction file.

        Raises
        ---------------
        `TypeError`:
            Raised when the acquired module, by means of `importlib`, is not a `ModuleType` object.
        """

        # cut the full reaction filepath to just the filename
        reaction_filename = reaction_filepath.split("\\")[-1]

        # set up the import functions to access the reaction file module
        spec = importlib.util.spec_from_file_location(reaction_filename, reaction_filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # ensure the loaded module is, in fact, a module type object
        if not isinstance(module, types.ModuleType):
            raise TypeError(
                "ERROR: Acquired Module is not a ModuleType! "
                "Reaction parameters cannot be accessed."
            )

        # get a list of all dictionary keys in the reaction file
        all_keys = [item for item in module.__dict__.keys() if not item.startswith("_")]

        # create a dictionary ot contain the reaction file parameters and their key identifiers
        output_params = {}

        # iterate through the key identifiers and add reaction parameters to the output dictionary
        for key in all_keys:
            # acquire the reaction parameter from the module
            parameter = module.__dict__[key]

            # ensure the parameter is not an imported module (because these are not required)
            if not isinstance(parameter, types.ModuleType):
                # add the parameter to the dictionary at the position designated by `key`
                output_params[key] = parameter

        return output_params

    def get_ni_label(self):
        """
        Method to obtain the names of all the reactants used in the experiment.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `labels` : `list`
            A list of the names of each reactant used in this reaction

        Raises
        ---------------
        None
        """

        labels = ['[{}]'.format(reactant) for reactant in self.reactants]

        return labels

    def get_ni_num(self):
        """
        Method to determine which chemicals are available for use.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `num_list` : `list`
            A list of the chemicals that are currently available (in hand)

        Raises
        ---------------
        None
        """

        # populate a list with the available chemicals
        num_list = []
        for i in range(self.cur_in_hand.shape[0]):
            num_list.append(self.cur_in_hand[i])

        return num_list

    def action_deconstruct(self, action):
        """
        Method to deconstruct the action and obtain the proper thermodynamic variables and reactant
        amounts that can be used in further methods.

        Parameters
        ---------------
        `action` : `list`
            A list of values specifying actions that are to be carried out.
            Each action varies from 0 to 1.

        Returns
        ---------------
        `temp_change` : `float`
            The change in temperature set to occur as given by the inputted action.
        `volume_change` : `float`
            The change in volume set to occur as given by the inputted action.
        `delta_n_array` : `np.array`
            An array containing the changes in molar amount of each reactant as given by the action.

        Raises
        ---------------
        None
        """

        # the first action parameter is the requested change in temperature;
        # action[0] = 0.0 indicates a temperature change of -dT
        # action[0] = 1.0 indicates a temperature change of +dT
        temp_change = 2.0 * (action[0] - 0.5) * self.dT

        # the second action parameter is the requested change in volume;
        # action[0] = 0.0 indicates a volume change of -dV
        # action[0] = 1.0 indicates a volume change of +dV
        volume_change = 2.0 * (action[1] - 0.5) * self.dV

        # the remaining action parameters indicate the proportion of
        # each reactant requested to be added to the reaction vessel;
        # action[2] = 1.0 indicates that all available amounts of reactant[0] are to be added
        # action[2] = 0.0 indicates that no amount of reactant[0] is to be added
        reactant_proportions = action[2:]

        # to determine the molar amount of each reactant to be added, we need
        # to know how much of each reactant is currently available "in hand";
        # we will create a variable to store the molar amounts of reactants to be added
        delta_n_array = np.zeros(len(self.cur_in_hand))

        # we will iterate through all the materials available in hand and use the proportions
        # to calculate the molar amount of each reactant that has been requested to be added
        for i, reactant_amount in enumerate(self.cur_in_hand):
            # obtain the reactant proportion from the action
            reactant_proportion = reactant_proportions[i]

            # define the amount of the current reactant that is available
            amount_available = reactant_amount

            # calculate the molar amount to be added (denoted delta_n)
            delta_n = reactant_proportion * amount_available

            # add the amount to be added to the delta_n_array for storage
            delta_n_array[i] = delta_n

        return temp_change, volume_change, delta_n_array

    def vessel_deconstruct(self, vessels):
        """
        Method to deconstruct and acquire key properties of the inputted vessel.
        Additionally, the n array is given the appropriate values.
        Notable properties include the temperature and volume.

        Parameters
        ---------------
        `vessels` : `vessel.Vessel`
            A vessel containing materials that are to be used in a series of reactions.

        Returns
        ---------------
        `temperature` : `float`
            The temperature of the inputted vessel.
        `volume` : `float`
            The volume of the inputted vessel.

        Raises
        ---------------
        None
        """

        # obtain the material dictionary
        material_dict = vessels._material_dict

        # unpack the material dictionary and add all the material amounts to the n array
        for i, material_name in enumerate(material_dict.keys()):
            if material_name in self.materials:
                material_amount = material_dict[material_name][1]
                self.n[i] = material_amount

        # acquire the vessel's temperature property
        temperature = vessels.get_temperature()

        # acquire the vessel's volume property
        volume = vessels.get_volume()

        return temperature, volume

    def perform_compatibility_check(self, action, vessels, n_steps, step_num):
        '''
        Method to ensure vessel and action being supplied from reaction bench is compatible
        with perform_action method.

        Parameters
        ---------------
        `action` : `np.array`
            An array containing elements describing the changes to be made, during the current
            step, to each modifiable thermodynamic variable and reactant used in the reaction.
        `vessels` : `vessel.Vessel`
            A vessel containing materials that are to be used in a series of reactions.
         `n_steps` : `int`
            The number of increments into which the action is split.
        `step_num` : `int`
            The number of steps completed by the agent

        Returns
        ---------------

        Raises
        ---------------
        `TypeError`:
            Raised when the acquired action from reaction bench is not of type np.ndarray.
        `TypeError`:
            Raised when the acquired vessel is not a `Vessel` type object.
        `TypeError`:
            Raised when n_steps is not an integer.
        None
        '''

        # ensure action is an np.ndarray
        if not isinstance(action, np.ndarray):
            raise TypeError('Invalid action array. Unable to proceed')

        # ensure the vessel class is a class type object
        if not isinstance(vessels, vessel.Vessel):
            raise TypeError('Invalid vessel class. Unable to proceed')

        # ensure that n_steps is an integer
        if not isinstance(n_steps, int):
            raise TypeError("Invalid 'n_steps' type. Unable to proceed")

        # ensure that materials from vessels compatible with reaction
        materials_dict_vessel = vessels.get_material_dict()
        materials_array = []
        for reactant in materials_dict_vessel:
            materials_array.append(reactant)
        if step_num == 1:
            # in the first iteration materials_array will only be filled with reactants (no products)
            assert self.reactants.sort() == materials_array.sort()
        else:
            # after the first iteration materials_array should be filled with products as well
            assert set(materials_array).issubset(set(self.materials))

        # ensure that solutes from vessels compatible with reaction
        solute_dict_vessel = vessels.get_solute_dict()
        solute_list = []
        for solute_dict in solute_dict_vessel.values():
            for key in solute_dict.keys():
                if key not in solute_list:
                    solute_list.append(key)

        assert self.solutes == solute_list

        # ensure that Tmin are the same
        assert self.Tmin==vessels.Tmin

        # ensure that Tmax are the same
        assert self.Tmax==vessels.Tmax

        #ensure that Vmin are the same
        assert self.Vmin==vessels.v_min

        # ensure that Tmax are the same
        assert self.Vmax == vessels.v_max

        # ensure that dt are the same
        assert self.dt == vessels.default_dt

    def update_vessel(self, temperature, volume):
        """
        Method to update the provided vessel object with materials from the reaction base
        and new thermodynamic variables.

        Parameters
        ---------------
        `temperature` : `float`
            The final temperature of the vessel after all reactions have been completed.
        `volume` : `float`
            The final temperature of the vessel after all reactions have been completed.

        Returns
        ---------------
        `new_vessel` : `float`
            The new vessel containing the final materials and thermodynamic variables.

        Raises
        ---------------
        None
        """

        # tabulate all the materials used and their new values
        new_material_dict = {}
        for i in range(self.n.shape[0]):
            material_name = self.materials[i]
            material_class = self.material_classes[i]
            amount = self.n[i]
            new_material_dict[material_name] = [material_class, amount, 'mol']

        # tabulate all the solutes and their values
        new_solute_dict = {}
        for mat, mat_class in zip(self.materials, self.material_classes):
            if mat not in self.solutes and mat_class()._solute:
                for i in range(self.initial_solutes.shape[0]):
                    solute_name = self.solutes[i]
                    solute_class = self.solute_classes[i]
                    amount = self.initial_solutes[i]

                    # create the new solute dictionary to be appended to a new vessel object
                    new_solute_dict[mat] = {solute_name: [solute_class, amount, 'mol']}

        # create a new vessel and update it with new data
        new_vessel = vessel.Vessel(
            'react_vessel',
            temperature=self.Ti,
            v_max=self.Vmax,
            v_min=self.Vmin,
            Tmax=self.Tmax,
            Tmin=self.Tmin,
            default_dt=self.dt
        )
        new_vessel.temperature = temperature
        new_vessel.volume = volume
        new_vessel._material_dict = new_material_dict
        new_vessel._solute_dict = new_solute_dict
        new_vessel.pressure = new_vessel.get_pressure()

        return new_vessel

    def reset(self, vessels):
        """
        Method to reset the environment and vessel back to its initial state.
        Empty the initial n array and reset the vessel's thermodynamic properties.
        Parameters
        ---------------
        `vessels` : `vessel.Vessel`
            A vessel containing initial materials.

        Returns
        ---------------
        `vessels` : `vessel.Vessel`
            A vessel that has been updated to have the proper thermodynamic properties.

        Raises
        ---------------
        None
        """

        # open the provided vessel to get the material and solute dictionaries
        material_dict = vessels.get_material_dict()
        solute_dict = vessels.get_solute_dict()

        # acquire the amounts of reactant materials
        for i, material_name in enumerate(material_dict.keys()):
            if material_name in self.reactants:
                self.initial_in_hand[i] = material_dict[material_name][1]

        # acquire the amounts of solute materials
        for i, solute_name in enumerate(solute_dict.keys()):
            if solute_name in self.solutes:
                self.initial_solutes[i] = solute_dict[solute_name][1]

        vessels = vessel.Vessel(
            'react_vessel',
            materials=material_dict,
            solutes=solute_dict,
            temperature=self.Ti,
            v_max=self.Vmax,
            v_min=self.Vmin,
            Tmax=self.Tmax,
            Tmin=self.Tmin,
            default_dt=self.dt
        )

        # ensure the entire initial materials in hand array is available
        self.cur_in_hand = 1.0 * self.initial_in_hand

        # set the n array to contain initial values
        for i, material_amount in enumerate(self.initial_in_hand):
            self.n[i] = material_amount

        # reset the vessel parameters to their original values as specified in the reaction file
        vessels.set_v_min(self.Vmin, unit="l")
        vessels.set_v_max(self.Vmax, unit="l")
        vessels.set_volume(self.Vi, unit="l", override=True)
        vessels.temperature = self.Ti
        vessels.Tmin = self.Tmin
        vessels.Tmax = self.Tmax
        vessels.default_dt = self.dt

        return vessels

    def update(self, conc, temp, volume, t, dt, n_steps):
        """
        Method to update the environment.
        This involves using reactants, generating products, and obtaining rewards.

        Parameters
        ---------------
        conc : np.array
            An array containing the concentrations of each material in the vessel.
        temp : np.float32
            The temperature of the system in Kelvin
        volume : np.float32
            The volume of the system in Litres
        dt : np.float32
            The time-step demarcating separate steps
        n_steps : np.int64
            The number of steps into which the action has been divided.

        Returns
        ---------------
        reward : np.float32
            The amount of the desired product created during the time-step

        Raises
        ---------------
        None
        """

        # set the intended vessel temperature in the differential equation module
        self.de.temp = temp

        # implement the differential equation solver
        if self.solver != 'newton':
            print("prev_conc")
            print(conc)
            print("prev_total_amount")
            print(self.n)

            new_conc = self._solver(self.de, (t, t + dt * n_steps), conc, method=self.solver).y[:, -1]
            for i in range(self.n.shape[0]):
                # convert back to moles and update the molar amount array
                self.n[i] = new_conc[i] * volume

            print("new_conc")
            print(new_conc)
            print("new_total_amount")
            print(self.n)
        else:
            print("prev_conc")
            print(conc)
            print("prev_total_amount")
            print(self.n)
            conc_change = self.de.run(conc, temp) * dt
            for i in range(self.n.shape[0]):
                # convert back to moles and update the molar amount array
                dn = conc_change[i] * volume
                self.n[i] += dn

            print("conc_change")
            print(conc_change)
            print("new_total_amount")
            print(self.n)

        # check the list of molar amounts and set negligible amounts to 0
        for i, amount in enumerate(self.n):
            if amount < self.threshold:
                self.n[i] = 0

    def perform_action(self, action, vessels: vessel.Vessel, t, n_steps, step_num):
        """
        Update the environment with processes defined in `action`.

        Parameters
        ---------------
        `action` : `np.array`
            An array containing elements describing the changes to be made, during the current
            step, to each modifiable thermodynamic variable and reactant used in the reaction.
        `vessels` : `vessel.Vessel`
            A vessel containing initial materials to be used in a series of reactions.
        `n_steps` : `int`
            The number of increments into which the action is split.

        Returns
        ---------------
        `vessels` : `vessel.Vessel`
            A vessel that has been updated to have the proper materials and
            thermodynamic properties.

        Raises
        ---------------
        None
        """

        self.perform_compatibility_check(
            action=action,
            vessels=vessels,
            n_steps=n_steps,
            step_num=step_num
        )

        # deconstruct the action
        temperature_change, volume_change, delta_n_array = self.action_deconstruct(action=action)

        # deconstruct the vessel: acquire the vessel temperature and volume and create the n array
        temperature, volume = self.vessel_deconstruct(vessels=vessels)
        current_volume = vessels.get_current_volume()[-1]

        # perform the complete action over a series of increments
        if self.solver != 'newton':
            n_steps = 1
        for __ in range(n_steps):
            # split the overall temperature change into increments,
            # ensure it does not exceed the maximal and minimal temperature values
            temperature += temperature_change / n_steps
            temperature = np.min(
                [
                    np.max([temperature, vessels.get_Tmin()]),
                    vessels.get_Tmax()
                ]
            )

            # split the overall volume change into increments,
            # ensure it does not exceed the maximal and minimal volume values
            volume += volume_change / n_steps
            volume = np.min(
                [
                    np.max([volume, vessels.get_min_volume()]),
                    vessels.get_max_volume()
                ]
            )

            # split the overall molar changes into increments
            for j, delta_n in enumerate(delta_n_array):
                dn = delta_n / n_steps
                self.n[j] += dn
                self.cur_in_hand[j] -= dn

                # set a penalty for trying to add unavailable material (tentatively set to 0)
                # reward -= 0

                # if the amount that is in hand is below a certain threshold set it to 0
                if self.cur_in_hand[j] < self.threshold:
                    self.cur_in_hand[j] = 0.0

            # perform the reaction and update the molar concentrations of the reactants and products
            self.update(
                self.n/current_volume,
                temperature,
                current_volume,
                t,
                vessels.get_defaultdt(),
                n_steps
            )

        # create a new reaction vessel containing the final materials
        vessels = self.update_vessel(temperature, volume)

        return vessels


    def plotting_step(self, t, tmax, vessels: vessel.Vessel):
        '''
        Set up a method to handle the acquisition of the necessary plotting parameters
        from an input vessel.

        Parameters
        ---------------
        `t` : `float`
            Current amount of time
        `tmax` : `float`
            Maximum time allowed
        `vessels` : `vessel.Vessel`
            A vessel containing methods to acquire the necessary parameters

        Returns
        ---------------
        `plot_data_state` : `list`
            A list containing the states to be plotted such as time,
            temperature, volume, pressure, and reactant amounts
        `plot_data_mol` : `list`
            A list containing the molar amounts of reactants and products
        `plot_data_concentration` : `list`
            A list containing the concentration of reactants and products

        Raises
        ---------------
        None
        '''

        # acquire the necessary parameters from the vessel
        T = vessels.get_temperature()
        V = vessels.get_volume()
        # V = 0.0025

        # create containers to hold the plotting parameters
        plot_data_state = [[], [], [], []]
        plot_data_mol = [[] for _ in range(self.n.shape[0])]
        plot_data_concentration = [[] for _ in range(self.n.shape[0])]

        # Record time data
        plot_data_state[0].append(t / tmax)

        # record temperature data
        Tmin = vessels.get_Tmin()
        Tmax = vessels.get_Tmax()
        plot_data_state[1].append((T - Tmin) / (Tmax - Tmin))

        # record volume data
        Vmin = vessels.get_min_volume()
        Vmax = vessels.get_max_volume()
        plot_data_state[2].append((V - Vmin) / (Vmax - Vmin))

        # record pressure data
        P = vessels.get_pressure()
        plot_data_state[3].append(
            P / vessels.get_pmax()
        )

        # calculate and record the molar concentrations of the reactants and products
        C = vessels.get_concentration(materials=self.materials)
        for j in range(self.n.shape[0]):
            plot_data_mol[j].append(self.n[j])
            plot_data_concentration[j].append(C[j])

        return plot_data_state, plot_data_mol, plot_data_concentration

    def plot_human_render(self, first_render, wave_data_dict, plot_data_state):
        '''
        Method to plot thermodynamic variables and spectral data.
        Plots a minimal amount of data for a 'surface-level'
        understanding of the information portrayed.

        Parameters
        ---------------
        `first_render` : `boolean`
            Indicates whether a plot has been already rendered
        `wave_data_dict` : `dict`
            A dictionary containing all the wave data
        `plot_data_state` : `list`
            A list containing the states we are going to plot such as time,
            temperature, volume, pressure, and reactant amounts

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        num_list = [plot_data_state[0][0],
                    plot_data_state[1][0],
                    plot_data_state[2][0],
                    plot_data_state[3][0]
                    ] + self.get_ni_num()

        reactants = []
        for reactant in self.reactants:
            name = reactant.replace("[", "").replace("]", "")
            short_name = name[0:5]
            reactants.append(short_name)

        label_list = ['t', 'T', 'V', 'P'] + reactants

        spectra_len = wave_data_dict["spectra_len"]
        absorb = wave_data_dict["absorb"]
        wave = wave_data_dict["wave"]
        wave_min = wave_data_dict["wave_min"]
        wave_max = wave_data_dict["wave_max"]

        if first_render:
            plt.close('all')
            plt.ion()

            # define a figure and several subplots
            self._plot_fig, self._plot_axs = plt.subplots(1, 2, figsize=(12, 6))

            # plot spectra
            self._plot_line1 = self._plot_axs[0].plot(
                wave,  # wave range from wave_min to wave_max
                absorb,  # absorb spectra
            )[0]
            self._plot_axs[0].set_xlim([wave_min, wave_max])
            self._plot_axs[0].set_ylim([0, 1.2])
            self._plot_axs[0].set_xlabel('Wavelength (nm)')
            self._plot_axs[0].set_ylabel('Absorbance')

            # bar chart displaying time, temperature, pressure,
            # and amounts of each species left available in hand
            __ = self._plot_axs[1].bar(
                range(len(num_list)),
                num_list,
                tick_label=label_list,
            )[0]
            self._plot_axs[1].set_ylim([0, 1])

            # draw the graph and show it
            self._plot_fig.canvas.draw()
            plt.show()

        # if the plot is already rendered, simply add the new data to it
        else:
            self._plot_line1.set_xdata(wave)
            self._plot_line1.set_ydata(absorb)

            # bar chart
            self._plot_axs[1].cla()
            __ = self._plot_axs[1].bar(
                range(len(num_list)),
                num_list,
                tick_label=label_list,
            )[0]
            self._plot_axs[1].set_ylim([0, 1])

            # draw the graph
            self._plot_fig.canvas.draw()
            plt.pause(0.000001)

    def plot_full_render(self, first_render, wave_data_dict, plot_data_state, plot_data_mol, plot_data_concentration,
                         n_steps, vessels: vessel.Vessel, step_num):
        '''
        Method to plot thermodynamic variables and spectral data.
        Plots a significant amount of data for a more in-depth
        understanding of the information portrayed.

        Parameters
        ---------------
        `first_render` : `boolean`
            Indicates whether a plot has been already rendered
        `wave_data_dict` : `dict`
            A dictionary containing all the wave data
        `plot_data_state` : `list`
            A list containing the states to be plotted such as time,
            temperature, volume, pressure, and reactant amounts
        `plot_data_mol` : `list`
            A list containing the molar amounts of reactants and products
        `plot_data_concentration` : `list`
            A list containing the concentration of reactants and products
        `n_steps` : `int`
            The number of increments into which the action is split
        `vessels` : `vessel.Vessel`
            A vessel containing methods to obtain thermodynamic data
        `step_num` : `int`
            The step number the environment is currently on

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        num_list = [plot_data_state[0][0],
                    plot_data_state[1][0],
                    plot_data_state[2][0],
                    plot_data_state[3][0]
                    ] + self.get_ni_num()

        reactants = []
        for reactant in self.reactants:
            name = reactant.replace("[", "").replace("]", "")
            short_name = name[0:5]
            reactants.append(short_name)

        label_list = ['t', 'T', 'V', 'P'] + reactants

        spectra_len = wave_data_dict["spectra_len"]
        absorb = wave_data_dict["absorb"]
        wave = wave_data_dict["wave"]
        wave_min = wave_data_dict["wave_min"]
        wave_max = wave_data_dict["wave_max"]

        peak = CharacterizationBench().get_spectra_peak(vessels, materials=self.materials)
        dash_spectra = CharacterizationBench().get_dash_line_spectra(vessels, materials=self.materials)

        # The first render is required to initialize the figure
        if first_render:
            plt.close('all')
            plt.ion()
            self._plot_fig, self._plot_axs = plt.subplots(2, 3, figsize=(24, 12))

            # Time vs. Molar_Amount graph ********************** Index: (0, 0)
            self._plot_lines_amount = np.zeros((self.n.shape[0],2,20))
            for i in range(self.n.shape[0]):
                self._plot_lines_amount[i][0][step_num - 1:] = (plot_data_state[0][0])
                self._plot_lines_amount[i][1][step_num - 1:] = (plot_data_mol[i][0])
                self._plot_axs[0, 0].plot(
                    self._plot_lines_amount[i][0],
                    self._plot_lines_amount[i][1],
                    label=self.materials[i]
                )
            self._plot_axs[0, 0].set_xlim([0.0, vessels.get_defaultdt() * n_steps])
            self._plot_axs[0, 0].set_ylim([0.0, np.mean(plot_data_mol)])
            self._plot_axs[0, 0].set_xlabel('Time (s)')
            self._plot_axs[0, 0].set_ylabel('Molar Amount (mol)')
            self._plot_axs[0, 0].legend()

            # Time vs. Molar_Concentration graph *************** Index: (0, 1)
            self._plot_lines_concentration = np.zeros((self.n.shape[0],2,20))
            for i in range(self.n.shape[0]):
                self._plot_lines_concentration[i][0][step_num - 1:] = (plot_data_state[0][0])
                self._plot_lines_concentration[i][1][step_num - 1:] = (plot_data_concentration[i][0])
                self._plot_axs[0, 1].plot(
                    self._plot_lines_concentration[i][0],
                    self._plot_lines_concentration[i][1],
                    label=self.materials[i]
                )
            self._plot_axs[0, 1].set_xlim([0.0, vessels.get_defaultdt() * n_steps])
            self._plot_axs[0, 1].set_ylim([0.0, np.mean(plot_data_concentration)])
            self._plot_axs[0, 1].set_xlabel('Time (s)')
            self._plot_axs[0, 1].set_ylabel('Molar Concentration (mol/L)')
            self._plot_axs[0, 1].legend()

            # Time vs. Temperature + Time vs. Volume graph *************** Index: (0, 2)
            self._plot_lines_temp = np.zeros((1, 2, 20))
            self._plot_lines_temp[0][0][step_num - 1:] = (plot_data_state[0][0])
            self._plot_lines_temp[0][1][step_num - 1:] = (plot_data_state[1][0])
            self._plot_axs[0, 2].plot(
                self._plot_lines_temp[0][0],
                self._plot_lines_temp[0][1],
                label='T'
            )
            self._plot_lines_vol = np.zeros((1, 2, 20))
            self._plot_lines_vol[0][0][step_num - 1:] = (plot_data_state[0][0])
            self._plot_lines_vol[0][1][step_num - 1:] = (plot_data_state[2][0])
            self._plot_axs[0, 2].plot(
                self._plot_lines_vol[0][0],
                self._plot_lines_vol[0][1],
                label='V'
            )
            self._plot_axs[0, 2].set_xlim([0.0, vessels.get_defaultdt() * n_steps])
            self._plot_axs[0, 2].set_ylim([0, 1.2])
            self._plot_axs[0, 2].set_xlabel('Time (s)')
            self._plot_axs[0, 2].set_ylabel('T and V (map to range [0, 1])')
            self._plot_axs[0, 2].legend()

            # Time vs. Pressure graph *************** Index: (1, 0)
            self._plot_lines_pressure = np.zeros((1, 2, 20))
            self._plot_lines_pressure[0][0][step_num - 1:] = (plot_data_state[0][0])
            self._plot_lines_pressure[0][1][step_num - 1:] = (plot_data_state[3][0])
            self._plot_axs[1, 0].plot(
                self._plot_lines_pressure[0][0],
                self._plot_lines_pressure[0][1],
                label='V'
            )
            self._plot_axs[1, 0].set_xlim([0.0, vessels.get_defaultdt() * n_steps])
            self._plot_axs[1, 0].set_ylim([0, vessels.get_pmax()])
            self._plot_axs[1, 0].set_xlabel('Time (s)')
            self._plot_axs[1, 0].set_ylabel('Pressure (kPa)')

            # Solid Spectra graph *************** Index: (1, 1)
            __ = self._plot_axs[1, 1].plot(
                wave,  # wavelength space (ranging from wave_min to wave_max)
                absorb  # absorb spectra
            )[0]

            # dash spectra
            for spectra in dash_spectra:
                self._plot_axs[1, 1].plot(wave, spectra, linestyle='dashed')

            # include the labelling when plotting spectral peaks
            for i in range(len(peak)):
                self._plot_axs[1, 1].scatter(peak[i][0], peak[i][1], label=peak[i][2])

            self._plot_axs[1, 1].set_xlim([wave_min, wave_max])
            self._plot_axs[1, 1].set_ylim([0, 1.2])
            self._plot_axs[1, 1].set_xlabel('Wavelength (nm)')
            self._plot_axs[1, 1].set_ylabel('Absorbance')
            self._plot_axs[1, 1].legend()

            # bar chart plotting the time, tempurature, pressure, and amount of each
            # species left in hand *************** Index: (1, 2)
            __ = self._plot_axs[1, 2].bar(
                range(len(num_list)),
                num_list,
                tick_label=label_list
            )[0]
            self._plot_axs[1, 2].set_ylim([0, 1])

            # draw and show the full graph
            self._plot_fig.canvas.draw()
            plt.show()

        # All other renders serve to update the existing figure with the new current state;
        # it is assumed that the above actions have already been carried out
        else:
            # set data for the Time vs. Molar_Amount graph *************** Index: (0, 0)
            curent_time = plot_data_state[0][-1]

            # update the lines data
            for i in range(self.n.shape[0]):
                # populate the lines amount array with the updated number of mols
                self._plot_lines_amount[i][0][step_num - 1:] = (plot_data_state[0][0])
                self._plot_lines_amount[i][1][step_num - 1:] = (plot_data_mol[i][0])
                self._plot_axs[0,0].lines[i].set_xdata(self._plot_lines_amount[i][0])
                self._plot_axs[0,0].lines[i].set_ydata(self._plot_lines_amount[i][1])

                # populate the lines concentration array with the updated number of concentration
                self._plot_lines_concentration[i][0][step_num - 1:] = (plot_data_state[0][0])
                self._plot_lines_concentration[i][1][step_num - 1:] = (plot_data_concentration[i][0])
                self._plot_axs[0, 1].lines[i].set_xdata(self._plot_lines_concentration[i][0])
                self._plot_axs[0, 1].lines[i].set_ydata(self._plot_lines_concentration[i][1])

            # reset each plot's x-limit because the latest action occurred at a greater time-value
            self._plot_axs[0, 0].set_xlim([0.0, curent_time])
            self._plot_axs[0, 1].set_xlim([0.0, curent_time])

            # reset each plot's y-limit in order to fit and show all materials on the graph
            self._plot_axs[0,0].set_ylim([0.0, np.mean(plot_data_mol)])
            self._plot_axs[0,1].set_ylim([0.0, np.mean(plot_data_concentration)])

            # set data for Time vs. Temp and Time vs. Volume graphs *************** Index: (0, 2)
            self._plot_lines_temp[0][0][step_num - 1:] = (plot_data_state[0][0])
            self._plot_lines_temp[0][1][step_num - 1:] = (plot_data_state[1][0])
            self._plot_lines_vol[0][0][step_num - 1:] = (plot_data_state[0][0])
            self._plot_lines_vol[0][1][step_num - 1:] = (plot_data_state[2][0])

            self._plot_axs[0,2].lines[0].set_xdata(self._plot_lines_temp[0][0])
            self._plot_axs[0,2].lines[0].set_ydata(self._plot_lines_temp[0][1])
            self._plot_axs[0,2].lines[1].set_xdata(self._plot_lines_vol[0][0])
            self._plot_axs[0,2].lines[1].set_ydata(self._plot_lines_vol[0][1])

            # reset xlimit since t extend
            self._plot_axs[0, 2].set_xlim([0.0, curent_time])

            # set data for Time vs. Pressure graph *************** Index: (1, 0)
            self._plot_lines_pressure[0][0][step_num - 1:] = (plot_data_state[0][0])
            self._plot_lines_pressure[0][1][step_num - 1:] = (plot_data_state[3][0])
            self._plot_axs[1, 0].lines[0].set_xdata(self._plot_lines_pressure[0][0])
            self._plot_axs[1, 0].lines[0].set_ydata(self._plot_lines_pressure[0][1])

            self._plot_axs[1, 0].set_xlim([0.0, curent_time])  # reset xlime since t extend
            self._plot_axs[1, 0].set_ylim([0.0, np.max(plot_data_state[3])*1.1])

            # reset the Solid Spectra graph
            self._plot_axs[1, 1].cla()
            __ = self._plot_axs[1, 1].plot(
                wave,  # wavelength space (ranging from wave_min to wave_max)
                absorb,  # absorb spectra
            )[0]

            # obtain the new dash spectra data
            for __, item in enumerate(dash_spectra):
                self._plot_axs[1, 1].plot(wave, item, linestyle='dashed')

            # reset the spectra peak labelling
            for i in range(len(peak)):
                self._plot_axs[1, 1].scatter(peak[i][0], peak[i][1], label=peak[i][2])

            self._plot_axs[1, 1].set_xlim([wave_min, wave_max])
            self._plot_axs[1, 1].set_ylim([0, 1.2])
            self._plot_axs[1, 1].set_xlabel('Wavelength (nm)')
            self._plot_axs[1, 1].set_ylabel('Absorbance')
            self._plot_axs[1, 1].legend()

            # bar chart
            self._plot_axs[1, 2].cla()
            __ = self._plot_axs[1, 2].bar(
                range(len(num_list)),
                num_list,
                tick_label=label_list,
            )[0]
            self._plot_axs[1, 2].set_ylim([0, 1])

            # re-draw the graph
            self._plot_fig.canvas.draw()
            plt.pause(0.000001)
