'''
Module to model all six Wurtz chlorine hydrocarbon reactions.

:title: wurtz_reaction.py

:author: Mitchell Shahen

:history: 22-06-2020
'''

# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=unsubscriptable-object

import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../") # allows module to access chemistrylab
from chemistrylab.ode_algorithms.spectra import diff_spectra as spec
from chemistrylab.reactions.get_reactions import convert_to_class
from chemistrylab.reactions.rate import Rates


R = 8.314462619

class _Reaction():

    def __init__(self, reactants, products, materials, desired, const_coef, exp_coef, rate_fn: Rates, solutes=None, overlap=False, nmax=None, max_mol=2, thresh=1e-8):
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

        self.name = "base_reaction"

        self.vessels = None
        self.threshold = thresh

        # Change this!!!
        # get the initial amounts of each reactant material, we need to change this!!!!
        initial_materials = np.zeros(len(reactants))
        for material in materials:
            if material["Material"] in reactants:
                index = reactants.index(material["Material"])
                initial_materials[index] = material["Initial"]

        self.initial_in_hand = initial_materials
        self.cur_in_hand = initial_materials

        self.n = initial_materials

        # Change this!!!
        # get the initial amount of each solute
        initial_solutes = np.zeros(len(solutes))
        for solute in solutes:
            if solute["Solute"] in solutes:
                index = solutes.index(solute["Solute"])
                initial_solutes[index] = solute["Initial"]
        self.initial_solutes = initial_solutes
        self.solute_labels = solutes

        # specify the desired material
        self.desired_material = desired
        self.reactants = reactants
        self.products = products
        self.materials = materials

        # Change this!!!
        # convert the reactants and products to their class object representations
        self.reactant_classes = convert_to_class(materials=reactants)
        self.product_classes = convert_to_class(materials=products)
        self.material_classes = convert_to_class(materials=materials)
        self.solute_classes = convert_to_class(materials=solutes)

        # define the maximum of each chemical allowed at one time (in mol)
        if not nmax:
            self.nmax = np.ones(len(self.material_classes))
        else:
            self.nmax = nmax

        # create labels for each of the chemicals involved
        self.labels = materials

        # store the class that calculates the reaction rates
        self.rate_fn = rate_fn
        self.const_coef = const_coef
        self.exp_coef = exp_coef
        # define the maximal number of moles available for any chemical
        self.max_mol = max_mol

        # define parameters for generating spectra
        self.params = [spec.S_1]*len(materials)

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

    def reset(self, n_init):
        '''
        Method to reset the environment back to its initial state.
        Populates two class instance attributes with initial data.

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

        # define a class instance attribute for the available chemicals
        self.cur_in_hand = 1.0 * self.initial_in_hand

        # define a class instance attribute for the amount of each chemical
        self.n = n_init

    def get_reaction_constants(self, temp):
        k = self.const_coef * np.exp((-1*self.exp_coef)/(R*temp))
        return k

    def get_rates(self, k, conc):
        return self.rate_fn.get_rates(k, conc)

    def get_conc_change(self, rates, dt):
        # define the mechanics of the reaction concentration change here
        return np.array([])

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
        conc = self.get_concentration(volume)
        k = self.get_reaction_constants(temp)
        rates = self.get_rates(k, conc)
        conc_change = self.get_conc_change(rates, dt)
        for i in range(self.n.shape[0]):
            # convert back to moles
            dn = conc_change[i] * volume
            self.n[i] += dn # update the molar amount array

            # check the list of molar amounts and set negligible amounts to 0
            for i, amount in enumerate(self.n):
                if amount < self.threshold:
                    self.n[i] = 0


    def get_total_pressure(self, V, T=300):
        '''
        Method to obtain the total pressure of all chemicals.

        Parameters
        ---------------
        V : np.float32
            The volume of the system in L
        T : np.float32 (default=300)
            The temperature of the system in Kelvin

        Returns
        ---------------
        P_total : np.float32
            The sum of the pressures of every chemical in the experiment

        Raises
        ---------------
        None
        '''

        # calculate the total pressure of all chemicals
        P_total = 0
        for i in range(self.n.shape[0]):
            P_total += self.n[i] * R * T / V

        return P_total

    def get_part_pressure(self, V, T=300):
        '''
        Method to obtain the individual pressure of each chemical.

        Parameters
        ---------------
        V : np.float32
            The volume of the system in L
        T : np.float32 (default=300)
            The temperature of the system in Kelvin

        Returns
        ---------------
        P : np.array
            An array of the pressures of each chemical in the experiment

        Raises
        ---------------
        None
        '''

        # create an array of all the pressures of each chemical individually
        P = np.zeros(self.n.shape[0], dtype=np.float32)
        for i in range(self.n.shape[0]):
            P[i] = self.n[i] * R * T / V

        return P



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

        # set the default reaction constants
        k = self.get_reaction_constants(temp)

        # set the default number of steps to evolve the system
        n_steps = n_steps

        # set arrays to keep track of the time and concentration
        t = np.zeros(n_steps, dtype=np.float32)
        conc = np.zeros((n_steps, 4), dtype=np.float32)

        # set initial concentration values
        conc[0] = conc[0] * self.initial_in_hand

        for i in range(1, n_steps):
            conc = self.get_concentration(volume)
            rates = self.get_rates(k, conc)
            conc_change = self.get_conc_change(rates, dt)

            # update plotting info
            conc[i] = conc[i-1] + conc_change
            t[i] = t[i-1] + dt

        plt.figure()

        for i in range(conc.shape[1]):
            plt.plot(t, conc[:, i], label=self.labels[i])

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
