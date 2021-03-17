'''
Module to model all six Wurtz chlorine hydrocarbon reactions.

:title: base_reaction.py

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
from chemistrylab.lab.de import De
from chemistrylab.chem_algorithms import util, vessel
import scipy
from scipy.integrate import solve_ivp


R = 8.314462619


class _Reaction:

    def __init__(self, initial_materials, initial_solutes, reactants, products, materials,
                 de: De, solver='newton', solutes=None, desired=None, overlap=False, nmax=None, max_mol=2, thresh=1e-8, Ti=0.0,
            Tmin=0.0, Tmax=0.0, dT=0.0, Vi=0.0, Vmin=0.0, Vmax=0.0, dV=0.0, dt=0):
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

        self.threshold = thresh

        # Change this!!!
        # get the initial amounts of each reactant material, we need to change this!!!!
        _initial_materials = np.zeros(len(reactants))
        for material in initial_materials:
            if material["Material"] in reactants:
                index = reactants.index(material["Material"])
                _initial_materials[index] = material["Initial"]

        self.initial_in_hand = _initial_materials
        self.cur_in_hand = _initial_materials

        self.n = _initial_materials

        # Change this!!!
        # get the initial amount of each solute
        _initial_solutes = np.zeros(len(solutes))
        for solute in initial_solutes:
            if solute["Solute"] in solutes:
                index = solutes.index(solute["Solute"])
                _initial_solutes[index] = solute["Initial"]
        self.initial_solutes = _initial_solutes

        # specify the materials
        self.desired = desired
        self.reactants = reactants
        self.products = products
        self.materials = materials
        self.solutes = solutes

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
        # store the class that calculates the reaction rates
        self.de = de
        # define the maximal number of moles available for any chemical
        self.max_mol = max_mol

        #define thermodynamic properties and volumetric properties for the reaction
        self.Ti = Ti
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.dT = dT
        self.Vi = Vi
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.dV = dV

        self.dt = dt
        self.solvers = {'RK45'}
        if solver in self.solvers:
            self.solver = solver
            self._solver = solve_ivp
        else:
            self.solver = 'newton'
            self._solver = None

        # define parameters for generating spectra
        # self.params = [spec.S_1]*len(materials)

        self.params = []
        for material in self.material_classes:
            if overlap:
                self.params.append(material().get_spectra_overlap())
            else:
                self.params.append(material().get_spectra_no_overlap())

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

    def step(self, action, vessels: vessel.Vessel, t, tmax, n_steps):
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

        T = vessels.get_temperature()
        V = vessels.get_volume()
        # the reward for this step is set to 0 initially

        # the first two action variables are changes to temperature and volume, respectively;
        # these changes need to be scaled according to the maximum allowed change, dT and dV;
        # action[0] = 1.0 gives a temperature change of dT, while action[0] = 0.0 corresponds to -dT
        # action[1] = 1.0 gives a temperature change of dV, while action[1] = 0.0 corresponds to -dV
        temperature_change = 2.0 * (action[0] - 0.5) * self.dT # in Kelvin
        volume_change = 2.0 * (action[1] - 0.5) * self.dV # in Litres

        # the remaining action variables are the proportions of reactants to be added
        add_reactant_proportions = action[2:]

        # set up a variable to contain the amount of change in each reactant
        delta_n_array = np.zeros(len(self.cur_in_hand))

        # scale the action value by the amount of reactant available
        for i, reactant_amount in enumerate(self.cur_in_hand):
            # determine how much of the available reactant is to be added
            proportion = add_reactant_proportions[i]

            # determine the amount of the reactant available
            amount_available = reactant_amount

            # determine the amount of reactant to add (in mol)
            delta_n = proportion * amount_available

            # add the overall change in materials to the array of molar amounts
            delta_n_array[i] = delta_n

        plot_data_state = [[], [], [], []]
        plot_data_mol = [[] for _ in range(self.n.shape[0])]
        plot_data_concentration = [[] for _ in range(self.n.shape[0])]
        if self.solver == 'newton':
            for i in range(n_steps):
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
                for i, delta_n in enumerate(delta_n_array):
                    dn = delta_n / n_steps
                    self.n[i] += dn
                    self.cur_in_hand[i] -= dn

                    # set a penalty for trying to add unavailable material (tentatively set to 0)
                    # reward -= 0

                    # if the amount that is in hand is below a certain threshold set it to 0
                    if self.cur_in_hand[i] < 1e-8:
                        self.cur_in_hand[i] = 0.0

                # perform the reaction and update the molar concentrations of the reactants and products
                self.update(
                    T,
                    V,
                    vessels.get_defaultdt(),
                    n_steps
                )
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
                P = self.get_total_pressure(V, T)
                plot_data_state[3].append(
                    P/vessels.get_pmax()
                )

                # calculate and record the molar concentrations of the reactants and products
                C = self.get_concentration(V)
                for j in range(self.n.shape[0]):
                    plot_data_mol[j].append(self.n[j])
                    plot_data_concentration[j].append(C[j])

                t += self.dt
            else:
                T += temperature_change
                T = np.min([
                    np.max([T, vessels.get_Tmin()]),
                    vessels.get_Tmax()
                ])

                # split the overall volume change into increments
                V += volume_change
                V = np.min([
                    np.max([V, vessels.get_min_volume()]),
                    vessels.get_max_volume()
                ])

                # split the overall molar changes into increments
                for i, delta_n in enumerate(delta_n_array):
                    dn = delta_n
                    self.n[i] += dn
                    self.cur_in_hand[i] -= dn

                    # set a penalty for trying to add unavailable material (tentatively set to 0)
                    # reward -= 0

                    # if the amount that is in hand is below a certain threshold set it to 0
                    if self.cur_in_hand[i] < 1e-8:
                        self.cur_in_hand[i] = 0.0

                # perform the reaction and update the molar concentrations of the reactants and products
                self.update(
                    T,
                    V,
                    vessels.get_defaultdt(),
                    n_steps
                )
                # Record time data
                plot_data_state[0].append(t)

                # record temperature data
                Tmin = vessels.get_Tmin()
                Tmax = vessels.get_Tmax()
                plot_data_state[1].append((T - Tmin) / (Tmax - Tmin))

                # record volume data
                Vmin = vessels.get_min_volume()
                Vmax = vessels.get_max_volume()
                plot_data_state[2].append((V - Vmin) / (Vmax - Vmin))

                # record pressure data
                P = self.get_total_pressure(V, T)
                plot_data_state[3].append(
                    P / vessels.get_pmax()
                )

                # calculate and record the molar concentrations of the reactants and products
                C = self.get_concentration(V)
                for j in range(self.n.shape[0]):
                    plot_data_mol[j].append(self.n[j])
                    plot_data_concentration[j].append(C[j])

                t += self.dt * n_steps


        # update the vessels variable
        vessels = self.update_vessel(vessels, T, V, self.get_total_pressure(V, T))

        return t, vessels, plot_data_state, plot_data_mol, plot_data_concentration

    def update(self, temp, volume, dt, n_steps):
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
        self.de.temp = temp

        if self.solver != 'newton':
            new_conc = self._solver(self.de, (0, dt*n_steps), conc, method=self.solver).y[:, -1]
            for i in range(self.n.shape[0]):
                # convert back to moles
                self.n[i] = new_conc[i] * volume  # update the molar amount array
        else:
            conc_change = self.de.run(conc, temp) * dt
            for i in range(self.n.shape[0]):
                # convert back to moles
                dn = conc_change[i] * volume
                self.n[i] += dn  # update the molar amount array
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

        # set the default number of steps to evolve the system
        n_steps = n_steps

        # set arrays to keep track of the time and concentration
        t = np.zeros(n_steps, dtype=np.float32)
        conc = np.zeros((n_steps, 4), dtype=np.float32)

        # set initial concentration values
        conc[0] = conc[0] * self.initial_in_hand

        for i in range(1, n_steps):
            conc_change = self.de.run(conc[i-1], temp)

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

    def update_vessel(self, vessels: vessel.Vessel, temperature, volume, pressure):
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