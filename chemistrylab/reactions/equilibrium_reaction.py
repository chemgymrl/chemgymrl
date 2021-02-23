import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("../../")
from chemistrylab.ode_algorithms.spectra import diff_spectra as spec
from chemistrylab.reactions.get_reactions import convert_to_class

# Reactions
# 1) A + B -> |C|
# 2) |C| -> A + B

R = 0.008314462618  # Gas constant (kPa * m**3 * mol**-1 * K**-1)

# constant for k = e^(-E/RT) for each reaction
# 1) A + B -> |C|
A1 = 2.0
E1 = 8.0

# 2) |C| -> A + B
A2 = 2.0
E2 = 10.0

REACTANTS = ['CuSO4*5H2O', 'H2O', 'CuSO4']
PRODUCTS = ['CuSO4*5H2O', 'H2O', 'CuSO4']
ALL_MATERIALS = ['CuSO4*5H2O', 'H2O', 'CuSO4']
SOLUTES = ['H2O']

class Reaction(object):
    def __init__(self, materials=None, solutes=None, desired="", overlap=False):
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
            [100.0 for __ in ALL_MATERIALS]
        )

        # create labels for each of the chemicals involved
        self.labels = ALL_MATERIALS

        # define a space to record all six reaction rates
        self.rate = np.zeros(3)

        # define the maximal number of moles available for any chemical
        self.max_mol = 100.0
        # Parameters to generate up to 3 Gaussian peaks per species
        self.params = []
        if overlap:
            # overlap spectra
            self.params.append(spec.S_6)  # spectra for A
            self.params.append(spec.S_7)  # spectra for B
            self.params.append(spec.S_3_3)  # spectra for C
        else:
            # no overlap spectra
            self.params.append(spec.S_1)  # spectra for A
            self.params.append(spec.S_3)  # spectra for B
            self.params.append(spec.S_5)  # spectra for C

    def get_ni_label(self):
        return REACTANTS

    def get_ni_num(self):
        num_list = []
        for i in range(self.cur_in_hand.shape[0]):
            num_list.append(self.cur_in_hand[i])
        return num_list

    # Reinitializes the reaction for reset purpose in main function
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

    # update the concentration of each chemical for a time step under tempurature T
    def update(self, T, V, dt):
        C = self.get_concentration(V)
        dC = np.zeros(C.shape[0])  # chenge of amount of A B C
        k1 = A1 * np.exp((-1 * E1) / (R * T))
        k2 = A2 * np.exp((-1 * E2) / (R * T))
        self.rate[0] = k1 * C[0] * C[1] * dt  # Rate of reaction 1
        self.rate[1] = k2 * C[2]
        dC[0] = -1.0 * self.rate[0] + self.rate[1] # change of A
        dC[1] = 1.0 * self.rate[0] + self.rate[1] # change of B
        dC[2] = 5.0 * self.rate[0] - self.rate[1] # change of C
        # Update concentration of each chemical
        for i in range(C.shape[0]):
            self.n[i] += dC[i] * V * 1000  # need convert V from m^3 to L
        d_reward = dC[2] * V * 1000  # Reward is change in mol of C
        return d_reward

    def get_total_pressure(self, V, T=300):
        P_total = 0
        for i in range(self.n.shape[0]):
            P_total += self.n[i] * R * T / V
        return P_total

    def get_part_pressure(self, V, T=300):
        P = np.zeros(self.n.shape[0], dtype=np.float32)
        for i in range(self.n.shape[0]):
            P[i] = self.n[i] * R * T / V
        return P

    def get_concentration(self, V=0.1):
        C = np.zeros(self.n.shape[0], dtype=np.float32)
        for i in range(self.n.shape[0]):
            C[i] = self.n[i] / (V * 1000)
        return C

    def get_spectra(self, V):
        # Initialize array for wavelength[0, 1] and absorbance
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
        absorb = np.zeros(x.shape[0], dtype=np.float32)
        C = self.get_concentration(V)
        for i in range(len(self.params)):  # for each species
            for j in range(self.params[i].shape[0]):  # for each peak
                for k in range(x.shape[0]):
                    absorb[k] += C[i] * self.params[i][j, 0] * np.exp(
                        -0.5 * ((x[k] - self.params[i][j, 1]) / self.params[i][j, 2]) ** 2.0)
                    # amount    *    height      *  decay rate
        # Maximum possible absorbance at any wavelength is 1.0
        absorb = np.clip(absorb, 0.0, 1.0)
        return absorb

    def get_spectra_peak(self, V):
        spectra_peak = []
        C = self.get_concentration(V)
        spectra_peak.append([self.params[0][:, 1] * 600 + 200, C[0] * self.params[0][:, 0], 'A'])
        spectra_peak.append([self.params[1][:, 1] * 600 + 200, C[1] * self.params[1][:, 0], 'B'])
        spectra_peak.append([self.params[2][:, 1] * 600 + 200, C[2] * self.params[2][:, 0], 'C'])
        return spectra_peak

    def get_dash_line_spectra(self, V):
        dash_spectra = []
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
        C = self.get_concentration(V)
        for i in range(len(self.params)):  # for each species
            each_absorb = np.zeros(x.shape[0], dtype=np.float32)
            for j in range(self.params[i].shape[0]):  # for each peak
                for k in range(x.shape[0]):
                    each_absorb[k] += C[i] * self.params[i][j, 0] * np.exp(
                        -0.5 * ((x[k] - self.params[i][j, 1]) / self.params[i][j, 2]) ** 2.0)
                    # amount    *    height      *  decay rate
            dash_spectra.append(each_absorb)
        return dash_spectra

    # plot the hard-coded graph for visualization
    def plot_graph(self):
        A = 20  # Initial A
        B = 100  # Initial B
        C = 20  # Initial C
        dt = 0.01  # Time step (s)
        k1 = A1 * np.exp((-1 * E1) / (R * 300))  # Rate of reaction 1 a function to T and P
        n_steps = 500  # Number of steps to evolve system
        t = np.zeros(n_steps, dtype=np.float32)  # Keep track of time for plotting
        conc = np.zeros((n_steps, 3), dtype=np.float32)  # Keep track of concentrations for plotting
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
        plt.xlim([t[0], t[-1]])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration (M)')
        plt.legend()
        plt.savefig('reaction_ex_0.pdf')
        plt.show()
        plt.close()
