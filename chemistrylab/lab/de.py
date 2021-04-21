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

This class is used to solve the chemical reaction(s).

:title: de.py

:author: Mark Baula, Nicholas Paquin, and Mitchell Shahen

:history: 26-03-2021

Used throughout this class are the following arrays:
    - rate_coef: is a num_reagents x num_reactions array of exponential coefficients that we use to calculate
    the rate of each reaction

    - exp_coef: is a num_reactions length vector which represents the exponential coefficients we use to calculate
    the rate constants

    - reaction_coef: in a num_reactions x num_materials numpy array
"""

import numpy as np
import copy

# specify the R constant
R = 8.314462619


class De:
    def __init__(self, stoich_coeff_arr: np.array, activ_energy_arr: np.array, conc_coeff_arr: np.array,
                 num_reagents: int):
        """
        Constructor class for differential equation solver

        Parameters
        ---------------
        `stoich_coeff_arr` : `np.array`
            An array of stoichiometric coefficients for each reaction that is to occur.
        `activ_energy_arr` : `np.array`
            An array including the activation energies of each reaction that is to occur.
        `conc_coeff_arr` : `np.array`
            An array including the coefficients of concentrations used to determine the rate of each reaction.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # acquire the necessary arrays
        self.stoich_coeff_arr = stoich_coeff_arr
        self.activ_energy_arr = activ_energy_arr
        self.conc_coeff_arr = conc_coeff_arr

        # specify the number of reagents being considered
        self.num_reagents = num_reagents

        # set the vessel temperature parameter
        self.temp = 0

    def __call__(self, t, conc):
        """
        a function that calculates the change in concentration given a current concentration
        remember to set the temperature before you call this function
        This function is mainly used with the scipy ODE solvers

        Parameters
        ---------------
        `conc` : `np.array`
            An array containing the initial concentrations of each material involved in the intended reactions.
        `t` : `np.float32`
            A necessary time-stamp.

        Returns
        ---------------
        `conc_change` : `np.array`
            An array of changes in concentration to each of the materials in the inputted conc array.

        Raises
        ---------------
        None
        """

        # acquire the change of concentrations array by executing the `run` function
        conc_change = self.run(conc, self.temp)

        return conc_change

    def run(self, conc, temp):
        """
        Method to run all the functions necessary to produce an array containing intended changes in concentration.

        Parameters
        ---------------
        `conc` : `np.array`
            An array containing the initial concentrations of each material involved in the intended reactions.
        `temp` : `np.array`
            The temperature of the vessel.

        Returns
        ---------------
        `conc_change` : `np.array`
            An array of changes in concentration to each of the materials in the inputted conc array.

        Raises
        ---------------
        None
        """

        # calculate all the reaction rate constants
        k = self.get_reaction_constants(temp, conc)

        # calculate the rates of each reaction
        rates = self.get_rates(k, conc)

        # calculate the changes in concentration
        conc_change = self.get_conc_change(rates)

        # ensure the concentration changes do not exceed limitations
        conc_change = self.conc_limit(conc_change, conc)
        return conc_change

    def get_reaction_constants(self, temp, conc):
        """
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
        `temperature` : `np.float32`
            The temperature of the vessel when the reaction(s) are taking place.
        `conc_arr` : `np.array`
            The array containing concentrations of each material set to take place in reaction(s).

        Returns
        ---------------
        `k` : `np.array`
            An array containing reaction rate constants for each reaction set to occur.

        Raises
        ---------------
        None
        """

        # acquire the activation energies and stoichiometric coefficients for each reaction
        activ_energy_arr = self.activ_energy_arr
        stoich_coeff_arr = self.stoich_coeff_arr

        # obtain the number of reactions set to occur
        n_reactions = activ_energy_arr.shape[0]

        # acquire the concentrations of the reactants from the full concentration array
        reactant_conc = conc[:self.num_reagents]

        # only non-zero reactant concentrations will be of use, so these are isolated
        non_zero_conc = np.array([conc for conc in reactant_conc if conc != 0.0])

        # we use the non-zero concentrations to define the aggregate concentration
        agg_conc = np.product(non_zero_conc)

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
            agg_conc_dim = len(non_zero_conc) if len(non_zero_conc) != 0 else 1

            # calculate the exponential needed to be applied to the aggregate concentration
            # to properly define the rate constant's pre-exponential factor
            exponent = (rate_constant_dim) / agg_conc_dim if agg_conc_dim != 0 else 1

            # apply the exponent to the aggregate rate constant to get the scaling factor
            scaling_factor = (abs(agg_conc)) ** exponent

            # add the scaling factor to the array of scaling factors
            scaling_arr[i] = scaling_factor

        # use the activation energy and scaling factor arrays to calculate the rate constants for each reaction
        assert 0 not in scaling_arr
        assert np.nan not in scaling_arr
        assert np.inf not in scaling_arr
        scaling_coefficients = 1 / scaling_arr
        k = scaling_coefficients * np.exp((-1.0 * activ_energy_arr) / (R * temp))

        return k

    def get_rates(self, k, conc):
        """
        Method to calculate the rates of each reaction set to occur.

        Parameters
        ---------------
        `k` : `np.array`
            An array of reaction rate constants for each of the intended reactions.
        `conc` : `np.array`
            An array containing the initial concentrations of each material involved in the intended reactions.

        Returns
        ---------------
        `rates` : `np.array`
            An array of each reaction set to occur.

        Raises
        ---------------
        None
        """

        # get the reaction rates by raising the concentrations of reactants to their stiochiometric coefficients
        rates = copy.deepcopy(k)
        for i in range(len(rates)):
            for k in range(self.num_reagents):
                rates[i] *= conc[k] ** self.stoich_coeff_arr[i][k]

        return rates

    def get_conc_change(self, rates):
        """
        Method to calculate the changes in concentration of each material.

        Parameters
        ---------------
        `rates` : `np.array`
            An array containing the rates of each reaction.

        Returns
        ---------------
        `conc_change` : `np.array`
            An array containing the changes in concentration to each of the materials in the inputted conc array.

        Raises
        ---------------
        None
        """

        # use the concentration coefficients and the rates of each reaction to calculate the changes in concentration
        conc_change = np.zeros(len(self.conc_coeff_arr))
        for i in range(len(conc_change)):
            for k in range(len(rates)):
                conc_change[i] += self.conc_coeff_arr[i][k]*rates[k]
        return conc_change

    def conc_limit(self, conc_change, conc):
        """
        This function makes sure that the change in concentration doesn't exceed the concentration of the reagents.

        Parameters
        ---------------
        `conc_change` : `np.array`
            An array containing the changes in concentration to each of the materials in the inputted conc array.
        `conc` : `np.array`
            An array containing the initial concentrations of each material involved in the intended reactions.

        Returns
        ---------------
        `conc_change` : `np.array`
            A validated array containing the changes in concentration to each of the materials.

        Raises
        ---------------
        None
        """

        # ensure that all the intended concentration changes do not exceed the initial available concentrations
        for i in range(self.num_reagents):
            conc_change[i] = np.max([conc_change[i], -1 * conc[i]])
        # print("reagent_conc")
        # print(conc[:self.num_reagents])
        # print("conc_change: before")
        # print(conc_change)
        # reactions = copy.deepcopy(self.conc_coeff_arr.T)
        # frac_array = np.zeros(len(reactions))
        # for i in range(self.num_reagents):
        #     for elem in self.conc_coeff_arr[i]:
        #         if elem < 0:
        #             frac_array += np.abs(elem)
        # produced = np.zeros(len(reactions))
        # for i, reaction in enumerate(reactions):
        #     for k in range(self.num_reagents):
        #         if reaction[k] != 0:
        #             if produced[i] == 0:
        #                 produced[i] = abs(conc_change[k] / frac_array[k])
        #             else:
        #                 produced[i] = min(abs(conc_change[k] / frac_array[k] ), produced[i])
        # print("produced")
        # print(produced)
        # for i in range(self.num_reagents, len(conc_change)):
        #     conc_change[i] = np.sum(self.conc_coeff_arr[i] * produced)
        # print("conc_change: after")
        # print("+++++++++++++++++++++++++++++++++")
        # print(conc_change)
        return conc_change
