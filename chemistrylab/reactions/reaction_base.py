'''
Module to model all six Wurtz chlorine hydrocarbon reactions.

:title: base_reaction.py

:author: Mitchell Shahen

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

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../") # allows module to access chemistrylab

from chemistrylab.reactions.get_reactions import convert_to_class
from chemistrylab.chem_algorithms import vessel

R = 8.314462619

class _Reaction:

    def __init__(self, reaction_file_identifier="", overlap=False):
        """
        Constructor class module for the Reaction class.

        Parameters
        ---------------
        `materials` : `list` (default=`None`)
            A list of dictionaries containing initial material names, classes, and amounts.
        `solutes` : `list` (default=`None`)
            A list of dictionaries containing initial solute names, classes, and amounts.
        `desired` : `str` (default="")
            A string indicating the name of the desired material.
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
        self.reactants = reaction_params["reactants"]
        self.products = reaction_params["products"]
        self.solutes = reaction_params["solutes"]
        self.desired = reaction_params["desired"]

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

        # define the maximal number of moles available for any chemical
        self.max_mol = 2.0

        # define the maximal number of moles of each chemical allowed at one time
        self.nmax = self.max_mol * np.ones(len(self.materials))

        # define parameters for generating spectra
        # self.params = [spec.S_1]*len(materials)

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
        Method to acquire the necessary reaction parameters and make them available.
        '''

        # cut the full reaction filepath to just the filename
        reaction_filename = reaction_filepath.split("\\")[-1]

        # cut the reaction filename to its basename (without extension)
        reaction_basename = reaction_filename.split("\\")[0]

        # set up the import functions to access the reaction file module
        spec = importlib.util.spec_from_file_location(reaction_basename, reaction_filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # acquire the necessary parameters from the module and add them to a dictionary
        output_params = {
            "reactants" : module.REACTANTS,
            "products" : module.PRODUCTS,
            "solutes" : module.SOLUTES,
            "desired" : module.DESIRED,
            "Ti" : module.Ti,
            "Vi" : module.Vi,
            "dt" : module.dt,
            "Tmin" : module.Tmin,
            "Tmax" : module.Tmax,
            "dT" : module.dT,
            "Vmin" : module.Vmin,
            "Vmax" : module.Vmax,
            "dV" : module.dV,
            "activ_energy_arr" : module.activ_energy_arr,
            "stoich_coeff_arr" : module.stoich_coeff_arr,
            "conc_coeff_arr" : module.conc_coeff_arr
        }

        return output_params

    def get_ni_label(self):
        '''
        Method to obtain the names of all the reactants used in the experiment.

        Parameters
        ---------------
        None

        Returns
        ---------------
        labels : list
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
        num_list : list
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
        V : np.float32 (default=0.1)
            The volume of the system in L

        Returns
        ---------------
        C : np.array
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

    def reset(self, vessels):
        '''
        Method to reset the environment back to its initial state.
        Empty the initial n array and open the vessel to determine
        the amounts of material initially available.

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

        # ensure the n array (which contains the amount of materials in use) is completely empty
        self.n = np.zeros(len(self.materials))

        initial_in_hand = np.zeros(len(self.reactants))
        initial_solutes = np.zeros(len(self.solutes))

        # open the provided vessel to get the material and solute dictionaries
        material_dict = vessels._material_dict
        solute_dict = vessels._solute_dict

        # acquire the amounts of reactant materials
        for i, material_name in enumerate(material_dict.keys()):
            if material_name in self.reactants:
                initial_in_hand[i] = material_dict[material_name][1]

        # acquire the amounts of solute materials
        for i, solute_name in enumerate(solute_dict.keys()):
            if solute_name in self.solutes:
                initial_solutes[i] = solute_dict[solute_name][1]

        # set the initial materials in hand array
        self.initial_in_hand = initial_in_hand

        # set the initial solutes in hand array
        self.initial_solutes = initial_solutes

        # ensure the entire initial materials in hand array is available
        self.cur_in_hand = 1.0 * initial_in_hand

        # reset the vessel parameters to their original values as specified in the reaction file
        vessels.set_v_min(self.Vmin, unit="l")
        vessels.set_v_max(self.Vmax, unit="l")
        vessels.set_volume(self.Vi, unit="l", override=True)
        vessels.temperature = self.Ti
        vessels.Tmin = self.Tmin
        vessels.Tmax = self.Tmax
        vessels.default_dt = self.dt

        return vessels

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

    def update_vessel(self, vessels: vessel.Vessel, temperature, volume, pressure):
        '''
        Method to update the provided vessel object with materials from the reaction base
        and new thermodynamic variables.
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
            default_dt=vessels.default_dt
        )
        new_vessel.temperature = temperature
        new_vessel._material_dict = new_material_dict
        new_vessel._solute_dict = new_solute_dict
        new_vessel.volume = volume
        new_vessel.pressure = pressure

        return new_vessel

    def update(self, temp, volume, dt):
        '''
        Method to update the environment.
        This involves using reactants, generating products, and obtaining rewards.

        Parameters
        ---------------
        T : np.float32
            The temperature of the system in Kelvin
        V : np.float32
            The volume of the system in Litres
        dt : np.float32
            The time-step demarcating separate steps

        Returns
        ---------------
        reward : np.float32
            The amount of the desired product created during the time-step

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

        Returns
        ---------------
        `state` : `np.array`
            An array containing updated values of all the thermodynamic variables,
            reactants, and spectra generated by the most recent step.
        `reward` : `float`
            The amount of the desired product that has been created in the most recent step.
        `done` : `boolean`
            Indicates if, upon completing the most recent step, all the required steps
            have been completed.
        `parameters` : `dict`
            Any additional parameters used/generated in the most recent step that need
            to be specified.

        Raises
        ---------------
        None
        '''

        # deconstruct the action
        temperature_change, volume_change, delta_n_array = self.action_deconstruct(action=action)

        # acquire the necessary thermodynamic parameters from the vessel
        T = vessels.get_temperature()
        V = vessels.get_volume()

        # perform the complete action over a series of increments
        for __ in range(n_steps):
            # split the overall temperature change into increments
            T += temperature_change / n_steps
            T = np.min([
                np.max([T, vessels.get_Tmin()]),
                vessels.get_Tmax()
            ])

            # split the overall volume change into increments
            V += volume_change / n_steps
            V = np.min([
                np.max([V, vessels.get_min_volume()]),
                vessels.get_max_volume()
            ])

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
                T,
                V,
                vessels.get_defaultdt()
            )

        # update the vessels variable
        vessels = self.update_vessel(vessels, T, V, vessels.get_pressure())

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

    def plotting_step(self, t, vessels: vessel.Vessel):
        '''
        Set up a method to handle the acquisition of the necessary plotting parameters
        from an input vessel.
        '''

        # acquire the necessary parameters from the vessel
        T = vessels.get_temperature()
        V = vessels.get_current_volume()

        # create containers to hold the plotting parameters
        plot_data_state = [[], [], [], []]
        plot_data_mol = [[] for _ in range(self.n.shape[0])]
        plot_data_concentration = [[] for _ in range(self.n.shape[0])]

        # Record time data
        plot_data_state[0].append(t)

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
        plot_data_state[3].append(
            P/vessels.get_pmax()
        )

        # calculate and record the molar concentrations of the reactants and products
        C = self.get_concentration(V)
        for j in range(self.n.shape[0]):
            plot_data_mol[j].append(self.n[j])
            plot_data_concentration[j].append(C[j])

        return plot_data_state, plot_data_mol, plot_data_concentration
