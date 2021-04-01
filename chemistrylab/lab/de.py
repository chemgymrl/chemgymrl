import numpy as np

"""
This class is used to solve the chemical reaction

rate_coef: is a num_reagents x num_reactions array of exponential coefficients that we use to calculate
the rate of each reaction

exp_coef: is a num_reactions length vector which represents the exponential coefficients we use to calculate
the rate constants

reaction_coef: in a num_reactions x num_materials numpy array

"""

class De:
    def __init__(self, stoich_coeff_arr: np.array, activ_energy_arr: np.array, conc_coeff_arr: np.array, num_reagents: int):
        self.stoich_coeff_arr = stoich_coeff_arr
        self.activ_energy_arr = activ_energy_arr
        self.conc_coeff_arr = conc_coeff_arr
        self.num_reagents = num_reagents
        self.temp = 0

    def __call__(self, t, conc):
        """
        a function that calculates the change in concentration given a current concentration
        remember to set the temperature before you call this function
        This function is mainly used with the scipy ODE solvers
        """
        conc_change = self.run(conc, self.temp)
        return conc_change

    def run(self, conc, temp):
        k = self.get_reaction_constants(temp, conc)
        rates = self.get_rates(k, conc)
        conc_change = self.get_conc_change(rates)
        conc_change = self.conc_limit(conc_change, conc)
        return conc_change

    def get_rates(self, k, conc):
        rate = conc[:self.num_reagents] ** self.stoich_coeff_arr
        rate = k * np.array([np.product(x) for x in rate])
        return rate

    def get_reaction_constants(self, temp, conc):
        R = 8.314462619
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
        reactant_conc = conc

        # we define a variable to contain the aggregate concentration value
        agg_conc = 1

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
            agg_conc_dim = len(non_zero_conc)

            # calculate the exponential needed to be applied to the aggregate concentration
            # to properly define the rate constant's pre-exponential factor
            exponent = (rate_constant_dim) / agg_conc_dim if agg_conc_dim != 0 else 1

            # apply the exponent to the aggregate rate constant to get the scaling factor
            scaling_factor = (abs(agg_conc)) ** exponent

            # add the scaling factor to the array of scaling factors
            scaling_arr[i] = scaling_factor

        # iterate through the activate energy and scaling factor arrays to
        # calculate the rate constants for each reaction
        A = 1/scaling_arr
        k = A * np.exp((-1.0 * activ_energy_arr) / (R * temp))

        return k

    def get_conc_change(self, rates):
        dC = np.array([np.sum(x) for x in self.conc_coeff_arr * rates])
        return dC

    def conc_limit(self, conc_change, conc):
        """
        This function makes sure that the change in concentration doesn't exceed the concentration of the reagents
        """
        for i in range(self.num_reagents):
            conc_change[i] = np.max([conc_change[i], -1*conc[i]])
        return conc_change

