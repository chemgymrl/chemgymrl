'''
Module to perform a series of reactions as specified by an accompanying reaction file.

:title: reaction_base.py

:author: Mark Baula, Nicholas Paquin, and Mitchell Shahen

:history: 22-06-2020
'''

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

sys.path.append("../../") # allows the module to access chemistrylab

from chemistrylab.reactions.get_reactions import convert_to_class
from chemistrylab.chem_algorithms import vessel
from chemistrylab.chem_algorithms import util

R = 8.314462619

class _Reaction:

    def __init__(self, reaction_file_identifier="", overlap=False):
        '''
        Constructor class module for the Reaction class.

        Parameters
        ---------------
        `materials` : `list` (default=`None`)
            A list of dictionaries containing initial material names, classes, and amounts.
        `solutes` : `list` (default=`None`)
            A list of dictionaries containing initial solute names, classes, and amounts.
        `desired` : `str` (default="")
            A string indicating the name of the desired material.
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
        '''

        # define a name/label for this reaction base class
        self.name = "base_reaction"

        # ensure the requested reaction file is found and accessible
        reaction_filepath = self._find_reaction_file(reaction_file=reaction_file_identifier)

        # acquire the necessary parameters from the reaction file
        reaction_params = self._get_reaction_params(reaction_filepath=reaction_filepath)

        # UNPACK THE REACTION PARAMETERS:
        # materials used and the desired material
        self.reactants = reaction_params['REACTANTS']
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

        # specify the full list of materials
        self.materials = self.reactants + self.products

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

        # define parameters for generating spectra
        self.params = []
        for material in self.material_classes:
            if overlap:
                self.params.append(material().get_spectra_overlap())
            else:
                self.params.append(material().get_spectra_no_overlap())

    @staticmethod
    def _find_reaction_file(reaction_file=""):
        '''
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
        '''

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
                reaction_file + ".py" # re-add the extension
            )
        else:
            raise IOError("ERROR: Requested Reaction File Not Found in Directory!")

        return output_reaction_file

    @staticmethod
    def _get_reaction_params(reaction_filepath=""):
        '''
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
            Raised when the acquired module, by means of `importlib`, is not a `ModuleType` object.hem available.
        '''

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
        '''
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
        '''

        labels = ['[{}]'.format(reactant) for reactant in self.reactants]

        return labels

    def get_ni_num(self):
        '''
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
        '''

        # populate a list with the available chemicals
        num_list = []
        for i in range(self.cur_in_hand.shape[0]):
            num_list.append(self.cur_in_hand[i])

        return num_list

    def get_concentration(self, V=0.1):
        '''
        Method to convert molar volume to concentration.

        Parameters
        ---------------
        `V` : `float` (default=0.1)
            The volume of the system in L

        Returns
        ---------------
        `C` : `np.array`
            An array of the concentrations (in mol/L) of each chemical in the experiment.

        Raises
        ---------------
        None
        '''

        # create an array containing the concentrations of each chemical
        C = np.zeros(self.n.shape[0], dtype=np.float32)
        for i in range(self.n.shape[0]):
            C[i] = self.n[i] / V

        return C

    def get_reaction_constants(self, temp, conc_arr):
        '''
        Method for calculating the reaction rate constants. The reaction rate for each reaction
        uses a reaction rate constant specific to that reaction because each reaction has a unique
        activation energy, E. The rate constant is given by k = A * e^((-1.0 * E) / (R * T)).

        The pre-exponential factor, A, ensures that the rate constant is of the proper units.
        The reaction rate must be of units: mol / (L * s), but uses various concentrations of
        reactant materials during it's calculation. Therefore, the rate constant must counter
        or normalize these concentrations to ensure the proper rate units.

        Therefore, A = 1 / S, for a scaling factor, S, which is calculated using the concentrations
        of all the reactants which are used in calculating any reaction rate (normalization).

        For example:
            Consider A + 2B --> C.
            We calculate the rate of this reaction as: r = k * (A ** 1) * (B ** 2).
            Therefore, the rate will have units of: [r] = [k] * mol**3 / L**3.
            Therefore, the units of k must be: [k] = L**2 / (mol**2 * s)
            Using aggregate concentrations of A and B, we define: k = A * e^((-1.0 * E) / (R * T)).
            The pre-exponential factor, A, defines the units of the rate constant.
            Therefore, the pre-exponential factor must be: A = 1 / ([A] * [B])**n
            For some value, n, that gets the proper rate constant units.
            If the concentrations of A and B are both non-zero, n = 1 gives the required units.
            If one of the concentrations of A or B are non-zero, n = 2 gives the required units.
            If both are zero, no reaction using A and B are available, so n = 1 by default.
            Note that n is determined by the stiochiometric coefficients of the rate as well.
            Therefore, we will have different scaling factors for different reactions.

        Parameters
        ---------------
        `temperature` : `float`
            The temperature of the vessel when the reaction(s) are taking place.
        `conc_arr` : `np.array`
            The array containing concentrations of each material set to take place in reaction(s).

        Returns
        ---------------
        `k_arr` : `np.array`
            An array containing reaction rate constants for each reaction set to occur.

        Raises
        ---------------
        None
        '''

        # acquire the activation energies and stoichiometric coefficients for each reaction
        activ_energy_arr = self.activ_energy_arr
        stoich_coeff_arr = self.stoich_coeff_arr

        # obtain the number of reactions set to occur
        n_reactions = activ_energy_arr.shape[0]

        # acquire the concentrations of the reactants from the full concentration array
        reactant_conc = [conc_arr[i] for i, __ in enumerate(self.reactants)]

        # we define a variable to contain the aggregate concentration value
        agg_conc = 1

        # only non-zero reactant concentrations will be of use, so these are isolated
        non_zero_conc = [conc for conc in reactant_conc if conc != 0.0]

        # we use the non-zero concentrations to define the aggregate concentration
        for conc in non_zero_conc:
            agg_conc *= conc

        # define arrays to hold the scaling factor for each reaction
        scaling_arr = np.zeros(n_reactions)

        # iterate through each row in the activation energy and stoichiometric coefficient arrays
        # to get the parameters required and calculate the scaling factor for each reaction
        for i, row in enumerate(stoich_coeff_arr):
            # get the dimensionality of the rate by summing the stoichiometric coefficients
            rate_exponential = sum(row)

            # get the dimensionality of the rate constant required to balance the rate exponential
            rate_constant_dim = rate_exponential - 1

            # get the dimensionality of the aggregate concentration
            agg_conc_dim = len(non_zero_conc)

            # calculate the exponential needed to be applied to the aggregate concentration
            # to properly define the rate constant's pre-exponential factor
            exponent = (rate_constant_dim) / agg_conc_dim if agg_conc_dim != 0 else 1

            # apply the exponent to the aggregate rate constant to get the scaling factor
            scaling_factor = (abs(agg_conc)) ** exponent

            # add the scaling factor to the array of scaling factors
            scaling_arr[i] = scaling_factor

        # set up an array to contain the rate constants for each reaction
        k_arr = np.zeros(n_reactions)

        # iterate through the activate energy and scaling factor arrays to
        # calculate the rate constants for each reaction
        for i, scaling_factor in enumerate(scaling_arr):
            # get the activation energy
            activ_energy = activ_energy_arr[i][0]

            # calculate the pre-exponential factor
            A = 1.0 / scaling_factor

            # calculate the rate constant
            k = A * np.exp((-1.0 * activ_energy) / (R * temp))

            # add the rate constant to the array of rate constants
            k_arr[i] = k

        return k_arr

    def get_rates(self, k_arr, conc_arr):
        '''
        Method to calculate the reaction rates for every reaction that is to be performed.
        The rates are calculated using the concentrations of reactants made available, their
        stoichiometric coefficients, and the rate constants for each reaction.

         Parameters
        ---------------
        `k_arr` : `np.array`
            An array containing reaction rate constants for each reaction set to occur.
        `conc_arr` : `bool`
            The array containing concentrations of each material set to take place in reaction(s).

        Returns
        ---------------
        `rates_arr` : `np.array`
            An array containing the rate at which each reaction is to take place.

        Raises
        ---------------
        None
        '''

        # acquire the array of stiochiometric coefficients for each reaction
        stoich_coeff_arr = self.stoich_coeff_arr

        # obtain the number of reactions set to occur (number of rows in the stoich_coeff_arr)
        n_reactions = stoich_coeff_arr.shape[0]

        # define an array to contain the rates for each reaction
        rates_arr = np.zeros(n_reactions)

        # acquire the concentrations of the reactants from the full concentration array
        reactant_conc = [conc_arr[i] for i, __ in enumerate(self.reactants)]

        # iterate through each row stoichiometric coefficients array, calculating each reaction rate
        for i, row in enumerate(stoich_coeff_arr):
            # get the rate constant
            k = k_arr[i]

            # set up a variable to contain the product of the concentrations
            conc_product = 1.0

            # iterate through the items in the current row to calculate the product of the
            # concentrations exponentiated using their stoichiometric coefficients
            for j, stoich_coeff in enumerate(row):
                # get the reactant concentration
                conc = reactant_conc[j]

                # calculate the exponentiated concentration and include it in the product
                conc_product *= conc ** stoich_coeff

            # multiply the concentration product and the rate constant to get the reaction rate
            rate = k * conc_product

            # add the reaction rate to the array of reaction rates
            rates_arr[i] = rate

        return rates_arr

    def get_conc_change(self, rates_arr, conc_arr, dt):
        '''
        Method to calculate the change_concentration array containing the changes in reactant and
        product concentrations as a result of performing a reaction.

         Parameters
        ---------------
        `rates_arr` : `np.array`
            An array containing the rate at which each reaction is to take place.
        `conc_arr` : `bool`
            The array containing concentrations of each material set to take place in reaction(s).

        Returns
        ---------------
        `change_conc_arr` : `np.array`
            An array containing the changes in concentration of each material that has taken
            place in the occurring reaction(s).

        Raises
        ---------------
        None
        '''

        # acquire the concentration coefficients array conatining the coefficients indicating the
        # relative proportions of materials being used up or created in each reaction
        conc_coeff_arr = self.conc_coeff_arr

        # get the number of materials involved in the reactions (same as len(self.materials))
        n_materials = conc_coeff_arr.shape[0]

        # set up an array to contain the changes in concentration
        change_conc_arr = np.zeros(n_materials)

        # iterate through the concentration coefficient array,
        # calculating the changes of each material
        for i, row in enumerate(conc_coeff_arr):
            # create a variable to contain the cumulative change of a material
            cumulative_change = 0

            # iterate through each coefficient in the row calculating
            # the change in concentration from each reaction
            for j, coefficient in enumerate(row):
                # calculate the concentration change
                conc_change = coefficient * rates_arr[j] * dt

                # add the concentration change to the cumulative change
                cumulative_change += conc_change

            # store the overall concentration change in the change concentration array
            change_conc_arr[i] = cumulative_change

        # acquire the concentrations of the reactants from the full concentration array
        reactant_conc = [conc_arr[i] for i, __ in enumerate(self.reactants)]

        # look through the changes to the reactant concentrations and make sure they
        # do not exceed the concentrations of reactants available
        for i, available_conc in enumerate(reactant_conc):
            # get the intended concentration change
            conc_change = change_conc_arr[i]

            # change the intended concentration to the available concentration, if necessary
            change_conc_arr[i] = np.max([conc_change, -1.0 * available_conc])

        return change_conc_arr

    def action_deconstruct(self, action):
        '''
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
        '''

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
        '''
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
        '''

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

    def update_vessel(self, temperature, volume):
        '''
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
        '''

        # tabulate all the materials used and their new values
        new_material_dict = {}
        for i in range(self.n.shape[0]):
            material_name = self.materials[i]
            material_class = self.material_classes[i]
            amount = self.n[i]
            new_material_dict[material_name] = [material_class, amount, 'mol']

        # tabulate all the solutes and their values
        new_solute_dict = {}
        for i in range(self.initial_solutes.shape[0]):
            solute_name = self.solutes[i]
            solute_class = self.solute_classes[i]
            amount = self.initial_solutes[i]

            # create the new solute dictionary to be appended to a new vessel object
            new_solute_dict[solute_name] = [solute_class, amount]

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
        '''
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
        '''

        # open the provided vessel to get the material and solute dictionaries
        material_dict = vessels._material_dict
        solute_dict = vessels._solute_dict

        # acquire the amounts of reactant materials
        for i, material_name in enumerate(material_dict.keys()):
            if material_name in self.reactants:
                self.initial_in_hand[i] = material_dict[material_name][1]

        # acquire the amounts of solute materials
        for i, solute_name in enumerate(solute_dict.keys()):
            if solute_name in self.solutes:
                self.initial_solutes[i] = solute_dict[solute_name][1]

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

    def update(self, temp, volume, dt):
        '''
        Method to update the environment.
        This involves using reactants, generating products, and obtaining rewards.

        Parameters
        ---------------
        `temp` : `float`
            The temperature of the system in Kelvin
        `volume` : `float`
            The volume of the system in Litres
        `dt` : `float`
            The time-step demarcating separate steps

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # get an array of the concentrations of each material in the n array
        conc_arr = self.get_concentration(volume)

        # calculate the reaction rate constants using the
        # vessel temperature and concentration array
        k_arr = self.get_reaction_constants(temp, conc_arr)

        # calculate the reaction rates using the reaction rate constants and concentration array
        rates_arr = self.get_rates(k_arr, conc_arr)

        # calculate the changes in concentration for each material in an array
        conc_change = self.get_conc_change(rates_arr, conc_arr, dt)

        # iterate through each material in the n array and update it's amount
        for i in range(self.n.shape[0]):
            dn = conc_change[i] * volume # convert back to moles
            self.n[i] += dn # update the molar amount array

        # check the list of molar amounts and set negligible amounts to 0
        for i, amount in enumerate(self.n):
            if amount < self.threshold:
                self.n[i] = 0

    def perform_action(self, action, vessels: vessel.Vessel, n_steps):
        '''
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
        '''

        # deconstruct the action
        temperature_change, volume_change, delta_n_array = self.action_deconstruct(action=action)

        # deconstruct the vessel: acquire the vessel temperature and volume and create the n array
        temperature, volume = self.vessel_deconstruct(vessels=vessels)

        # perform the complete action over a series of increments
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
                temperature,
                volume,
                vessels.get_defaultdt()
            )

        # create a new reaction vessel containing the final materials
        vessels = self.update_vessel(temperature, volume)

        return vessels

    def get_spectra(self, V):
        '''
        Class method to generate total spectral data using a guassian decay.

        Parameters
        ---------------
        V : np.float32
            The volume of the system in Litres

        Returns
        ---------------
        absorb : np.array
            An array of the total absorption data of every chemical in the experiment

        Raises
        ---------------
        None
        '''

        # convert the volume in litres to the volume in m**3
        V = V / 1000

        # set the wavelength space
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)

        # define an array to contain absorption data
        absorb = np.zeros(x.shape[0], dtype=np.float32)

        # obtain the concentration array
        C = self.get_concentration(V)

        # iterate through the spectral parameters in self.params and the wavelength space
        for i, item in enumerate(self.params):
            for j in range(item.shape[0]):
                for k in range(x.shape[0]):
                    amount = C[i]
                    height = item[j, 0]
                    decay_rate = np.exp(
                        -0.5 * (
                            (x[k] - self.params[i][j, 1]) / self.params[i][j, 2]
                        ) ** 2.0
                    )
                    if decay_rate < 1e-30:
                        decay_rate = 0
                    absorb[k] += amount * height * decay_rate

        # absorption must be between 0 and 1
        absorb = np.clip(absorb, 0.0, 1.0)

        return absorb

    def get_spectra_peak(self, V):
        '''
        Method to populate a list with the spectral peak of each chemical.

        Parameters
        ---------------
        V : np.float32
            The volume of the system in litres.

        Returns
        ---------------
        spectra_peak : list
            A list of parameters specifying the peak of the spectra for each chemical

        Raises
        ---------------
        None
        '''

        # get the concentration of each chemical
        C = self.get_concentration(V)

        # create a list of the spectral peak of each chemical
        spectra_peak = []
        for i, material in enumerate(self.materials):
            spectra_peak.append([
                self.params[i][:, 1] * 600 + 200,
                C[i] * self.params[i][:, 0],
                material
            ])
        return spectra_peak

    def get_dash_line_spectra(self, V):
        '''
        Module to generate each individual spectral dataset using gaussian decay.

        Parameters
        ---------------
        V : np.float32
            The volume of the system in Litres

        Returns
        ---------------
        dash_spectra : list
            A list of all the spectral data of each chemical

        Raises
        ---------------
        None
        '''

        dash_spectra = []
        C = self.get_concentration(V)

        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)

        for i, item in enumerate(self.params):
            each_absorb = np.zeros(x.shape[0], dtype=np.float32)
            for j in range(item.shape[0]):
                for k in range(x.shape[0]):
                    amount = C[i]
                    height = item[j, 0]
                    decay_rate = np.exp(
                        -0.5 * (
                            (x[k] - self.params[i][j, 1]) / self.params[i][j, 2]
                        ) ** 2.0
                    )
                    each_absorb += amount * height * decay_rate
            dash_spectra.append(each_absorb)

        return dash_spectra

    def plot_graph(self, temp=300, volume=1, pressure=101.325, dt=0.01, n_steps=500, save_name=None):
        '''
        Method to plot the concentration, pressure, temperature, time, and spectral data.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # get the concentration array
        conc = self.get_concentration(volume)

        # set the default reaction constants
        k = self.get_reaction_constants(temp, conc)

        # set arrays to keep track of the time and concentration
        t = np.zeros(n_steps, dtype=np.float32)
        conc = np.zeros((n_steps, 4), dtype=np.float32)

        # set initial concentration values
        conc[0] = conc[0] * self.initial_in_hand

        for i in range(1, n_steps):
            conc = self.get_concentration(volume)
            rates = self.get_rates(k, conc)
            conc_change = self.get_conc_change(rates, conc, dt)

            # update plotting info
            conc[i] = conc[i-1] + conc_change
            t[i] = t[i-1] + dt

        plt.figure()

        for i in range(conc.shape[1]):
            plt.plot(t, conc[:, i], label=self.materials[i])

        # set plotting parameters
        plt.xlim([t[0], t[-1]])
        plt.ylim([0.0, 2.0])
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration (M)')
        plt.legend()
        if save_name:
            plt.savefig(save_name)
        plt.show()
        plt.close()

    def plotting_step(self, t, tmax, vessels: vessel.Vessel):
        '''
        Set up a method to handle the acquisition of the necessary plotting parameters
        from an input vessel.
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
        plot_data_state[0].append(t/tmax)

        # record temperature data
        Tmin = vessels.get_Tmin()
        Tmax = vessels.get_Tmax()
        plot_data_state[1].append((T - Tmin)/(Tmax - Tmin))

        # record volume data
        Vmin = vessels.get_min_volume()
        Vmax = vessels.get_max_volume()
        plot_data_state[2].append((V - Vmin)/(Vmax - Vmin))

        # record pressure data
        P = vessels.get_pressure()
        # P = 1175.6600956686177
        plot_data_state[3].append(
            P/vessels.get_pmax()
        )

        # calculate and record the molar concentrations of the reactants and products
        C = self.get_concentration(V)
        for j in range(self.n.shape[0]):
            plot_data_mol[j].append(self.n[j])
            plot_data_concentration[j].append(C[j])

        return plot_data_state, plot_data_mol, plot_data_concentration

    def plot_human_render(self, first_render, wave_data_dict, plot_data_state):

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

    def plot_full_render(self, first_render, wave_data_dict, plot_data_state, plot_data_mol, plot_data_concentration, n_steps, vessels: vessel.Vessel):

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

        peak = self.get_spectra_peak(vessels.get_volume())
        dash_spectra = self.get_dash_line_spectra(vessels.get_volume())

        # The first render is required to initialize the figure
        if first_render:
            plt.close('all')
            plt.ion()
            self._plot_fig, self._plot_axs = plt.subplots(2, 3, figsize=(24, 12))

            # Time vs. Molar_Amount graph ********************** Index: (0, 0)
            self._plot_lines_amount = []
            for i in range(self.n.shape[0]):
                self._plot_lines_amount.append(
                    self._plot_axs[0, 0].plot(
                        plot_data_state[0],  # time
                        plot_data_mol[i],  # mol of species C[i]
                        label=self.materials[i]  # names of species C[i]
                    )[0]
                )
            self._plot_axs[0, 0].set_xlim([0.0, vessels.get_defaultdt() * n_steps])
            self._plot_axs[0, 0].set_ylim([0.0, np.max(self.nmax)])
            self._plot_axs[0, 0].set_xlabel('Time (s)')
            self._plot_axs[0, 0].set_ylabel('Molar Amount (mol)')
            self._plot_axs[0, 0].legend()

            # Time vs. Molar_Concentration graph *************** Index: (0, 1)
            self._plot_lines_concentration = []
            for i in range(self.n.shape[0]):
                self._plot_lines_concentration.append(
                    self._plot_axs[0, 1].plot(
                        plot_data_state[0],  # time
                        plot_data_concentration[i],  # concentrations of species C[i]
                        label=self.materials[i]  # names of species C[i]
                    )[0]
                )
            self._plot_axs[0, 1].set_xlim([0.0, vessels.get_defaultdt() * n_steps])
            self._plot_axs[0, 1].set_ylim([
                0.0, np.max(self.nmax) / (vessels.get_min_volume())
            ])
            self._plot_axs[0, 1].set_xlabel('Time (s)')
            self._plot_axs[0, 1].set_ylabel('Molar Concentration (mol/L)')
            self._plot_axs[0, 1].legend()

            # Time vs. Temperature + Time vs. Volume graph *************** Index: (0, 2)
            self._plot_line_1 = self._plot_axs[0, 2].plot(
                plot_data_state[0],  # time
                plot_data_state[1],  # Temperature
                label='T'
            )[0]
            self._plot_line_2 = self._plot_axs[0, 2].plot(
                plot_data_state[0],  # time
                plot_data_state[2],  # Volume
                label='V'
            )[0]
            self._plot_axs[0, 2].set_xlim([0.0, vessels.get_defaultdt() * n_steps])
            self._plot_axs[0, 2].set_ylim([0, 1])
            self._plot_axs[0, 2].set_xlabel('Time (s)')
            self._plot_axs[0, 2].set_ylabel('T and V (map to range [0, 1])')
            self._plot_axs[0, 2].legend()

            # Time vs. Pressure graph *************** Index: (1, 0)
            self._plot_line_3 = self._plot_axs[1, 0].plot(
                plot_data_state[0],  # time
                plot_data_state[3]  # Pressure
            )[0]
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
            for i in range(self.n.shape[0]):
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
            for i in range(self.n.shape[0]):
                self._plot_lines_amount[i].set_xdata(plot_data_state[0])
                self._plot_lines_amount[i].set_ydata(plot_data_mol[i])

                # set data for the Time vs. Molar_Concentration graph *************** Index: (0, 1)
                self._plot_lines_concentration[i].set_xdata(plot_data_state[0])
                self._plot_lines_concentration[i].set_ydata(plot_data_concentration[i])

            # reset each plot's x-limit because the latest action occurred at a greater time-value
            self._plot_axs[0, 0].set_xlim([0.0, curent_time])
            self._plot_axs[0, 1].set_xlim([0.0, curent_time])

            # set data for Time vs. Temp and Time vs. Volume graphs *************** Index: (0, 2)
            self._plot_line_1.set_xdata(plot_data_state[0])
            self._plot_line_1.set_ydata(plot_data_state[1])
            self._plot_line_2.set_xdata(plot_data_state[0])
            self._plot_line_2.set_ydata(plot_data_state[2])
            self._plot_axs[0, 2].set_xlim([0.0, curent_time])  # reset xlime since t extend

            # set data for Time vs. Pressure graph *************** Index: (1, 0)
            self._plot_line_3.set_xdata(plot_data_state[0])
            self._plot_line_3.set_ydata(plot_data_state[3])
            self._plot_axs[1, 0].set_xlim([0.0, curent_time])  # reset xlime since t extend

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
            for i in range(self.n.shape[0]):
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