import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../../../") # allows module to access chemistrylab
from chemistrylab.ode_algorithms.spectra import diff_spectra as spec
from chemistrylab.reactions.get_reactions import convert_to_class

# Reactions
# 1) A -> |B| + C

R = 0.008314462618  # Gas constant (kPa * m**3 * mol**-1 * K**-1)

# constant for k = e^(-E/RT) for each reaction
# 1) A -> |B| + C
A1 = 2.0
E1 = 8.0

REACTANTS = ["NaCl"]
PRODUCTS = ['Na', 'Cl']
ALL_MATERIALS = REACTANTS + PRODUCTS
SOLUTES = ['H2O']
class Reaction():
    '''
    Class describing the 6 Wurtz reactions.
    '''

    def __init__(self, materials=None, solutes=None, desired="", overlap=False):
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
        `overlap` : `bool` (default=`False`)
            Indicate if the spectral plot includes overlapping plots.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        self.name = "demo_reaction"

        # get the initial amounts of each reactant material
        initial_materials = np.zeros(len(REACTANTS))
        for material in materials:
            if material["Material"] in REACTANTS:
                index = REACTANTS.index(material["Material"])
                initial_materials[index] = material["Initial"]
        self.initial_in_hand = initial_materials

        # get the initial amount of each solute
        initial_solutes = np.zeros(len(SOLUTES))
        for solute in solutes:
            if solute["Solute"] in SOLUTES:
                index = SOLUTES.index(solute["Solute"])
                initial_solutes[index] = solute["Initial"]
        self.initial_solutes = initial_solutes
        self.solute_labels = SOLUTES

        # specify the desired material
        self.desired_material = desired
        self.reactants = REACTANTS
        # convert the reactants and products to their class object representations
        self.reactant_classes = convert_to_class(materials=REACTANTS)
        self.product_classes = convert_to_class(materials=PRODUCTS)
        self.material_classes = convert_to_class(materials=ALL_MATERIALS)
        self.solute_classes = convert_to_class(materials=SOLUTES)

        # define the maximum of each chemical allowed at one time (in mol)
        self.nmax = np.array(
            [1.0 for __ in ALL_MATERIALS]
        )

        # create labels for each of the chemicals involved
        self.labels = ALL_MATERIALS

        # define a space to record all six reaction rates
        self.rate = np.zeros(6)

        # define the maximal number of moles available for any chemical
        self.max_mol = 2.0

        # define parameters for generating spectra
        self.params = []
        if overlap:
            self.params.append(spec.S_3_3)  # spectra for NaCl
            self.params.append(spec.S_6) # spectra for the Na
            self.params.append(spec.S_7) # spectra for the Cl
        else:
            self.params.append(spec.S_8) # spectra for NaCl
            self.params.append(spec.S_1) # spectra for the Na
            self.params.append(spec.S_3) # spectra for the Cl

    @staticmethod
    def get_ni_label():
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

        labels = ['[{}]'.format(reactant) for reactant in REACTANTS]

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

    def update(self, T, V, dt):
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
            The time-step demarcating steps

        Returns
        ---------------
        reward : np.float32
            The amount of the desired product created during the time-step

        Raises
        ---------------
        None
        '''

        # obtain the concentration (all concentrations are in mol/m**3)
        C = self.get_concentration(V)

        # define a space to contain the changes in concentration to each chemical
        dC = np.zeros(self.n.shape[0])

        # define the reaction constant for each reaction
        k1 = A1 * np.exp((-1 * E1)/(R * T))

        # define the rate of each reaction
        self.rate[0] = k1 * C[0] * dt

        # calculate and store the changes in concentration of each chemical
        dC[0] = -1*self.rate[0] # change in NaCl
        dC[1] = 1*self.rate[0] # change in Na
        dC[2] = 1*self.rate[0] # change in Cl

        # update the concentrations of each chemical
        for i in range(self.n.shape[0]):
            # convert back to moles
            # Note: concentration is in mol/m**3 and V is in L
            # moles = moles/m**3 * L * 0.001m**3/L
            dn = dC[i] * V * 0.001
            self.n[i] += dn # update the molar amount array

        # calculate the reward (new molar amount of the desired chemical, if present)
        d_reward = 0
        if self.desired_material in ALL_MATERIALS:
            index = ALL_MATERIALS.index(self.desired_material)
            d_reward = dC[index] * V * 0.001

        return d_reward

    def get_total_pressure(self, V, T=300):
        '''
        Method to obtain the total pressure of all chemicals.

        Parameters
        ---------------
        V : np.float32
            The volume of the system in Litres
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
            The volume of the system in Litres
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
            The volume of the system in Litres

        Returns
        ---------------
        C : np.array
            An array of the concentrations (in mol/m**3) of each chemical in the experiment.

        Raises
        ---------------
        None
        '''

        # create an array containing the concentrations of each chemical
        C = np.zeros(self.n.shape[0], dtype=np.float32)
        for i in range(self.n.shape[0]):
            C[i] = self.n[i] / (V * 0.001)

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
            The volume of the system.

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
        spectra_peak.append([self.params[0][:, 1] * 600 + 200, C[0] * self.params[0][:, 0], 'A'])
        spectra_peak.append([self.params[1][:, 1] * 600 + 200, C[1] * self.params[1][:, 0], 'B'])
        spectra_peak.append([self.params[2][:, 1] * 600 + 200, C[2] * self.params[2][:, 0], 'C'])

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

    def plot_graph(self):
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

        # set initial values for each chemical
        A = 1.0
        B = 0.0
        C = 0.0

        # set the default thermodynamic variables
        T = 300.0
        P = 101.325
        dt = 0.01

        # set the default reaction constants
        k1 = T / 300.0
        k2 = P / 300.0

        # set the default number of steps to evolve the system
        n_steps = 500

        # set arrays to keep track of the time and concentration
        t = np.zeros(n_steps, dtype=np.float32)
        conc = np.zeros((n_steps, 4), dtype=np.float32)

        # set initial concentration values
        conc[0, 0] = 1.0 * A
        conc[0, 1] = 1.0 * B
        conc[0, 2] = 1.0 * C

        for i in range(1, n_steps):
            # Calculate rate for this time step
            r1 = k1 * A * B
            # Update concentrations
            A += -1.0 * r1 * dt
            B += -1.0 * r1 * dt
            C += 1.0 * r1 * dt
            # Update plotting info
            conc[i, 0] = 1.0 * A
            conc[i, 1] = 1.0 * B
            conc[i, 2] = 1.0 * C
            t[i] = t[i - 1] + dt

        plt.figure()

        for i in range(conc.shape[1]):
            plt.plot(t, conc[:, i], label=self.labels[i])

        # set plotting parameters
        plt.xlim([t[0], t[-1]])
        plt.ylim([0.0, 2.0])
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration (M)')
        plt.legend()
        plt.savefig('reaction_ex_7.pdf')
        plt.show()
        plt.close()
