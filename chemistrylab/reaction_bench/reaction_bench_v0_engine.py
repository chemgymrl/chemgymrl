'''
Reaction Bench Engine Class

:title: reaction_bench_v0_engine

:author: Chris Beeler and Mitchell Shahen

:history: 21-05-2020
'''

# pylint: disable=attribute-defined-outside-init
# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=unsubscriptable-object
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position

import numpy as np
import gym
import gym.spaces
import matplotlib.pyplot as plt
import pickle
import os
import sys

sys.path.append("../../") # to access chemistrylab
sys.path.append("../reactions/") # to access all reactions
from chemistrylab.chem_algorithms import vessel
from chemistrylab.chem_algorithms.logger import Logger
from chemistrylab.reactions.wurtz_reaction import Reaction

R = 0.008314462618 # Gas constant (kPa * m**3 * mol**-1 * K**-1)
wave_max = 800
wave_min = 200

class ReactionBenchEnv(gym.Env):
    '''
    Class to define elements of an engine to represent a reaction.
    '''

    def __init__(
            self,
            reaction=Reaction,
            materials=None,
            solutes=None,
            desired="",
            n_steps=50,
            dt=0.01,
            Ti=300,
            Tmin=250,
            Tmax=500,
            dT=50,
            Vi=0.002,
            Vmin=0.001,
            Vmax=0.005,
            dV=0.0005,
            overlap=False,
            vessel_path=None
    ):
        '''
        Constructor class method to pass thermodynamic variables to class methods.

        Parameters
        ---------------
        `reaction` : `class`
            A reaction class containing various methods that
            properly define a reaction to be carried out.
        `materials` : `list` (default=`None`)
            A list of dictionaries including initial material names, classes, and amounts.
        `solutes` : `list` (default=`None`)
            A list of dictionaries including initial solute names, classes, and amounts.
        `desired` : `str` (default="")
            A string indicating the desired output material.
        `n_steps` : `int` (default=`50`)
            The number of time steps to be taken during each action.
        `dt` : `float` (default=`0.01`)
            The amount of time taken in each time step.
        `Ti` : `int` (default=`300`)
            The initial temperature of the system in Kelvin.
        `Tmin` : `int` (default=`250`)
            The minimal temperature of the system in Kelvin.
        `Tmax` : `int` (default=`500`)
            The maximal temperature of the system in Kelvin.
        `dT` : `int` (default=`50`)
            The maximal allowed temperature change, in Kelvin, for a single action.
        `Vi` : `float` (default=`0.002`)
            The initial volume of the system in litres.
        `Vmin` : `float` (default=`0.001`)
            The minimal volume of the system in litres.
        `Vmax` : `float` (default=`0.005`)
            The maximal volume of the system in litres.
        `dV` : `float` (default=`0.0005`)
            The maximal allowed volume change, in litres, for a single action.
        `overlap` : `boolean` (default=`False`)
            Indicates if the spectral plots show overlapping signatures.
        `vessel_path` : `str` (default=`None`)
            A string indicating the path to a vessel intended to be loaded into this module.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        self.name = "experiment_0"
        self.step_num = 1

        self.n_steps = n_steps # Time steps per action
        self.dt = dt # Time step of reaction (s)
        self.tmax = n_steps * dt * 20 # Maximum time (s) (20 is the registered max_episode_steps)
        self.Ti = Ti # Initial temperature (K)
        self.Tmin = Tmin # Minimum temperature of the system (K)
        self.Tmax = Tmax # Maximum temperature of the system (K)
        self.dT = dT # Maximum change in temperature per action (K)

        self.Vi = Vi # Initial Volume (m**3)
        self.Vmin = Vmin # Minimum Volume of the system (m**3)
        self.Vmax = Vmax # Maximum Volume of the system (m**3)
        self.dV = dV # Maximum change in Volume per action (m**3)

        # initialize the reaction
        self.reaction = reaction(
            overlap=overlap,
            materials=materials,
            solutes=solutes,
            desired=desired
        )

        # set the maximum pressure
        Pmax = self.reaction.max_mol * R * Tmax / Vmin
        self.Pmax = Pmax

        # initialize vessels by providing a empty default vessel or loading an existing saved vessel
        self.n_init = np.zeros(self.reaction.nmax.shape[0], dtype=np.float32)
        self.vessel_path = vessel_path
        if vessel_path == None:
            self.vessels = vessel.Vessel(
                'default',
                temperature=Ti,
                p_max=Pmax,
                v_max=Vmax * 1000,
                v_min=Vmin * 1000,
                Tmax=Tmax,
                Tmin=Tmin,
                default_dt=dt
            )
        else:
            with open(vessel_path, 'rb') as handle:
                b = pickle.load(handle)
                self.vessels = b

            for i in range(self.n_init.shape[0]):
                material_name = self.reaction.labels[i]
                material_class = self.reaction.material_classes[i]
                self.n_init[i] = self.vessels._material_dict[material_name][1]

        # reset the inputted reaction before performing any steps
        self.reaction.reset(n_init=self.n_init)

        # Defining action and observation space for OpenAI Gym framework
        # shape[0] is the change in amount of each reactant
        # + 2 is for the change in T and V
        # all actions range between 0 () and 1 ()
        # 0 = no reactant added and T, V are decreased by the maximum amount (-dT and -dV)
        # 1 = all reactant added and T, V are increased by the maximum amount (dT and dV)
        act_low = np.zeros(
            self.reaction.initial_in_hand.shape[0] + 2,
            dtype=np.float32
        )
        act_high = np.ones(
            self.reaction.initial_in_hand.shape[0] + 2,
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=act_low, high=act_high)

        # this an array denoting spectral signatures (varying
        # between 0.0 and 1.0) for a wide range of wavelengths
        absorb = self.reaction.get_spectra(Vi)

        # Observations have several attributes
        # + 4 indicates state variables time, temperature, volume, and pressure
        # initial_in_hand.shape[0] indicates the amount of each reactant
        # absorb.shape[0] indicates the numerous spectral signatures from get_spectra
        obs_low = np.zeros(
            self.reaction.initial_in_hand.shape[0] + 4 + absorb.shape[0],
            dtype=np.float32
        )
        obs_high = np.ones(
            self.reaction.initial_in_hand.shape[0] + 4 + absorb.shape[0],
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high)

        # Reset the environment upon calling the class
        self.reset()

        # initialize the logger
        # log_file = os.path.join(os.getcwd(), "log-v0")
        # self.logging = Logger(log_file=log_file, log_format="default", print_messages=True)
        # self.logging.initialize()
        # self.logging.add_log(message="Initialized Logging in {} for {}".format(self.__class__, self.reaction.name))

    def _update_state(self):
        '''
        Method to update the state vector with the current time, temperature, volume,
        pressure, the amounts of each reactant, and spectral parameters.

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

        absorb = self.reaction.get_spectra(self.V)

        # create an array to contain all state variables
        self.state = np.zeros(
            4 + # time T V P
            self.reaction.initial_in_hand.shape[0] + # reactants
            absorb.shape[0], # spectra
            dtype=np.float32
        )

        # populate the state array with updated state variables
        # state[0] = time
        self.state[0] = self.t / self.tmax

        # state[1] = temperature
        Tmin = self.vessels.get_Tmin()
        Tmax = self.vessels.get_Tmax()
        self.state[1] = (self.T - Tmin) / (Tmax - Tmin)

        # state[2] = volume
        Vmin = self.vessels.get_min_volume()
        Vmax = self.vessels.get_max_volume()
        self.state[2] = (self.V - Vmin) / (Vmax - Vmin)

        # state[3] = pressure
        total_pressure = self.reaction.get_total_pressure(self.V, self.T)
        Pmax = self.vessels.get_pmax()
        self.state[3] = total_pressure / Pmax

        # remaining state variables pertain to reactant amounts and spectra
        for i in range(self.reaction.initial_in_hand.shape[0]):
            self.state[i+4] = self.reaction.n[i] / self.reaction.nmax[i]
        for i in range(absorb.shape[0]):
            self.state[i + self.reaction.initial_in_hand.shape[0] + 4] = absorb[i]

    def _update_vessel(self):
        '''
        Method to update the information stored in the vessel upon completing a step.
        
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

        # get the temperature and pressure from the state variables
        temperature = self.state[1]
        volume = self.state[2]
        pressure = self.state[3]

        # tabulate all the materials used and their new values
        new_material_dict = {}
        for i in range(self.reaction.n.shape[0]):
            material_name = self.reaction.labels[i]
            material_class = self.reaction.material_classes[i]
            amount = self.reaction.n[i]
            new_material_dict[material_name] = [material_class, amount]

        # tabulate all the solutes and their values
        new_solute_dict = {}
        for i in range(self.reaction.initial_solutes.shape[0]):
            solute_name = self.reaction.solute_labels[i]
            solute_class = self.reaction.solute_classes[i]
            amount = self.reaction.initial_solutes[i]
            new_solute_dict[solute_name] = [solute_class, amount]

        # create a new vessel and update it with new data
        new_vessel = vessel.Vessel(
            'new',
            temperature=self.Ti,
            p_max=self.Pmax,
            v_max=self.Vmax * 1000,
            v_min=self.Vmin * 1000,
            Tmax=self.Tmax,
            Tmin=self.Tmin,
            default_dt=self.dt
        )
        new_vessel.temperature = temperature
        new_vessel._material_dict = new_material_dict
        new_vessel._solute_dict = new_solute_dict
        new_vessel.volume = volume
        new_vessel.pressure = pressure

        # replace the old vessel with the updated version
        self.vessels = new_vessel

    def _save_vessel(self):
        '''
        Method to save a vessel as a pickle file.
        
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

        # if a vessel path was provided when the environment was initialized, use it for saving vessels
        if isinstance(self.vessel_path, str):
            open_file = self.vessel_path
        # otherwise, save the vessels in the current working directory
        else:
            file_directory = os.getcwd()
            filename = "vessel_{}.pickle".format(self.name)
            open_file = os.path.join(file_directory, filename)

        # delete any existing vessel files to ensure the vessel is saved as intended
        if os.path.exists(open_file):
            os.remove(open_file)

        # open the intended vessel file and save the vessel as a pickle file
        with open(open_file, 'wb') as vessel_file:
            pickle.dump(self.vessels, vessel_file)

    def reset(self):
        '''
        Class method to reinitialize the environment by setting
        all thermodynamic variables to their initial values.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `state` : `np.array`
            A numpy array of the state values restored to their initial values.

        Raises
        ---------------
        None
        '''

        self._first_render = True
        self.done = False

        # reset the modifiable state variables
        self.t = 0.0
        self.T = 1.0 * self.vessels.get_temperature()
        self.V = 1.0 * self.Vi

        # reinitialize the reaction class
        self.reaction.reset(n_init=self.n_init)

        # save the necessary temperatures,  volumes, and pressure
        Ti = self.vessels.get_temperature()
        Tmin = self.vessels.get_Tmin()
        Tmax = self.vessels.get_Tmax()
        Vi = self.Vi
        Vmin = self.vessels.get_min_volume()
        Vmax = self.vessels.get_max_volume()
        total_pressure = self.reaction.get_total_pressure(self.V, self.T)

        # populate the state with the above variables
        self.plot_data_state = [
            [0.0],
            [(Ti - Tmin) / (Tmax - Tmin)],
            [(Vi - Vmin) / (Vmax - Vmin)],
            [total_pressure]
        ]
        self.plot_data_mol = []
        self.plot_data_concentration = []

        # [0] is time, [1] is Tempurature, [2:] is each species
        C = self.reaction.get_concentration(self.V)
        for i in range(self.reaction.nmax.shape[0]):
            self.plot_data_mol.append([self.reaction.n[i]])
            self.plot_data_concentration.append([C[i]])

        # update the state variables with the initial values
        self._update_state()

        # update the vessel variable
        self._update_vessel()

        return self.state

    def step(self, action):
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

        # the reward for this step is set to 0
        reward = 0.0

        for i, reactant in enumerate(self.reaction.cur_in_hand):
            # Actions [1:] add reactants to system, 0.0 = 0%, 1.0 = 100%
            # If action [i] is larger than the amount of the current reactant, all is added
            if all([
                    action[i+2] > reactant,
                    reactant != 0
            ]):
                # in the case of over-use we apply a penalty unless no reactants are left to add
                reward -= 0.1 * (action[i+2] - reactant)

            dn = np.min([action[i+2], reactant])
            self.reaction.n[i] += dn
            self.reaction.cur_in_hand[i] -= dn

        for i in range(self.n_steps):
            # Action [0] changes temperature by, 0.0 = -dT, 0.5 = 0, 1.0 = +dT (all in Kelvin)
            # Temperature change is linear between frames
            self.T += 2.0 * self.dT * (action[0] - 0.5) / self.n_steps
            self.T = np.min([
                np.max([self.T, self.vessels.get_Tmin()]),
                self.vessels.get_Tmax()
            ])

            # action[1] changes volume by: 0.0 = -dV, 0.5 = 0, 1.0 = dV (all in Litres)
            self.V += 2.0 * self.dV * (action[1] - 0.5) / self.n_steps
            self.V = np.min([
                np.max([self.V, self.vessels.get_min_volume()]),
                self.vessels.get_max_volume()
            ])
            d_reward = self.reaction.update(
                self.T,
                self.V,
                self.vessels.get_defaultdt()
            )
            self.t += self.dt # Increase time by time step
            reward += d_reward # Reward is change in concentration of desired material

            # Record time data
            self.plot_data_state[0].append(self.t)

            # record temperature data
            Tmin = self.vessels.get_Tmin()
            Tmax = self.vessels.get_Tmax()
            self.plot_data_state[1].append((self.T - Tmin)/(Tmax - Tmin))

            # record volume data
            Vmin = self.vessels.get_min_volume()
            Vmax = self.vessels.get_max_volume()
            self.plot_data_state[2].append((self.V - Vmin)/(Vmax - Vmin))

            # record pressure data
            self.plot_data_state[3].append(
                self.reaction.get_total_pressure(self.V, self.T)
            )

            C = self.reaction.get_concentration(self.V)
            for j in range(self.reaction.n.shape[0]):
                self.plot_data_mol[j].append(self.reaction.n[j])
                self.plot_data_concentration[j].append(C[j])

        # update the state variables with the changes during the last step
        self._update_state()

        # update the vessels variable
        self._update_vessel()

        # ---------- ADD ---------- #
        # add option to save or print intermediary vessel

        # save the vessel when the final step is complete
        if any([self.done, self.step_num == 20]):
            self._save_vessel()

        # update the step counter
        self.step_num += 1

        return self.state, reward, self.done, {}

    def render(self, model='human'):
        '''
        Method to specify the type of plot rendering.

        Parameters
        ---------------
        `model` : `str` (default="human")
            Indicates the type of rendering model to use when generating the plots.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        if model == 'human':
            self.human_render()
        elif model == 'full':
            self.full_render()
        else:
            print("Incorrect Render Model: {}".format(model))

    def human_render(self, model='plot'):
        '''
        Method to plot thermodynamic variables and spectral data.
        Plots a minimal amount of data for a 'surface-level'
        understanding of the information portrayed.

        Parameters
        ---------------
        `model` : `str` (default="plot")
            Indicates the type of graphic to render.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # Wavelength array and array length for plotting
        spectra_len = self.state.shape[0] - 4 - self.reaction.initial_in_hand.shape[0]
        absorb = self.state[4 + self.reaction.initial_in_hand.shape[0]:]

        # set a space for all spectral data to exist in
        wave = np.linspace(
            wave_min,
            wave_max,
            spectra_len,
            endpoint=True,
            dtype=np.float32
        )

        # set a list of state variables and their labels
        num_list = [
            self.state[0],
            self.state[1],
            self.state[2],
            self.state[3]
        ] + self.reaction.get_ni_num()
        label_list = ['t', 'T', 'V', 'P'] + self.reaction.get_ni_label()

        # define and render an empty plot if none such yet exists
        if self._first_render:
            plt.close('all')
            plt.ion()

            # define a figure and several subplots
            self._plot_fig, self._plot_axs = plt.subplots(1, 2, figsize=(12, 6))

            # plot spectra
            self._plot_line1 = self._plot_axs[0].plot(
                wave, # wave range from wave_min to wave_max
                absorb, # absorb spectra
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

            self._first_render = False

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

    def full_render(self, model='plot'):
        '''
        Method to plot thermodynamic variables and spectral data.
        Plots a significant amount of data for a more in-depth
        understanding of the information portrayed.

        Parameters
        ---------------
        `model` : `str` (default="plot")
            Indicates the type of graphic to render.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # define the length of the spectral data array and the array itself
        spectra_len = self.state.shape[0] - 4 - self.reaction.initial_in_hand.shape[0]
        absorb = self.state[4 + self.reaction.initial_in_hand.shape[0]:]

        # set a space for all spectral data to exist in
        wave = np.linspace(
            wave_min,
            wave_max,
            spectra_len,
            endpoint=True,
            dtype=np.float32
        )

        # obtain lists of the numbers and labels representing each
        num_list = [
            self.state[0],
            self.state[1],
            self.state[2],
            self.state[3]
        ] + self.reaction.get_ni_num()
        label_list = ['t', 'T', 'V', 'P'] + self.reaction.get_ni_label()

        # get the spectral data peak and dashed spectral lines
        peak = self.reaction.get_spectra_peak(self.V)
        dash_spectra = self.reaction.get_dash_line_spectra(self.V)

        # The first render is required to initialize the figure
        if self._first_render:
            plt.close('all')
            plt.ion()
            self._plot_fig, self._plot_axs = plt.subplots(2, 3, figsize=(24, 12))

            # Time vs. Molar_Amount graph ********************** Index: (0, 0)
            self._plot_lines_amount = []
            for i in range(self.reaction.n.shape[0]):
                self._plot_lines_amount.append(
                    self._plot_axs[0, 0].plot(
                        self.plot_data_state[0], # time
                        self.plot_data_mol[i], # mol of species C[i]
                        label=self.reaction.labels[i] # names of species C[i]
                    )[0]
                )
            self._plot_axs[0, 0].set_xlim([0.0, self.vessels.get_defaultdt() * self.n_steps])
            self._plot_axs[0, 0].set_ylim([0.0, np.max(self.reaction.nmax)])
            self._plot_axs[0, 0].set_xlabel('Time (s)')
            self._plot_axs[0, 0].set_ylabel('Molar Amount (mol)')
            self._plot_axs[0, 0].legend()

            # Time vs. Molar_Concentration graph *************** Index: (0, 1)
            self._plot_lines_concentration = []
            for i in range(self.reaction.n.shape[0]):
                self._plot_lines_concentration.append(
                    self._plot_axs[0, 1].plot(
                        self.plot_data_state[0], # time
                        self.plot_data_concentration[i], # concentrations of species C[i]
                        label=self.reaction.labels[i] # names of species C[i]
                    )[0]
                )
            self._plot_axs[0, 1].set_xlim([0.0, self.vessels.get_defaultdt() * self.n_steps])
            self._plot_axs[0, 1].set_ylim([
                0.0, np.max(self.reaction.nmax)/(self.vessels.get_min_volume() * 1000)
            ])
            self._plot_axs[0, 1].set_xlabel('Time (s)')
            self._plot_axs[0, 1].set_ylabel('Molar Concentration (mol/L)')
            self._plot_axs[0, 1].legend()

            # Time vs. Temperature + Time vs. Volume graph *************** Index: (0, 2)
            self._plot_line_1 = self._plot_axs[0, 2].plot(
                self.plot_data_state[0], # time
                self.plot_data_state[1], # Temperature
                label='T'
            )[0]
            self._plot_line_2 = self._plot_axs[0, 2].plot(
                self.plot_data_state[0], # time
                self.plot_data_state[2], # Volume
                label='V'
            )[0]
            self._plot_axs[0, 2].set_xlim([0.0, self.vessels.get_defaultdt() * self.n_steps])
            self._plot_axs[0, 2].set_ylim([0, 1])
            self._plot_axs[0, 2].set_xlabel('Time (s)')
            self._plot_axs[0, 2].set_ylabel('T and V (map to range [0, 1])')
            self._plot_axs[0, 2].legend()

            # Time vs. Pressure graph *************** Index: (1, 0)
            self._plot_line_3 = self._plot_axs[1, 0].plot(
                self.plot_data_state[0], # time
                self.plot_data_state[3] # Pressure
            )[0]
            self._plot_axs[1, 0].set_xlim([0.0, self.vessels.get_defaultdt() * self.n_steps])
            self._plot_axs[1, 0].set_ylim([0, self.vessels.get_pmax()])
            self._plot_axs[1, 0].set_xlabel('Time (s)')
            self._plot_axs[1, 0].set_ylabel('Pressure (kPa)')

            # Solid Spectra graph *************** Index: (1, 1)
            __ = self._plot_axs[1, 1].plot(
                wave, # wavelength space (ranging from wave_min to wave_max)
                absorb # absorb spectra
            )[0]

            # dash spectra
            for spectra in dash_spectra:
                self._plot_axs[1, 1].plot(wave, spectra, linestyle='dashed')

            # include the labelling when plotting spectral peaks
            for i in range(self.reaction.n.shape[0]):
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
            self._first_render = False

        # All other renders serve to update the existing figure with the new current state;
        # it is assumed that the above actions have already been carried out
        else:
            # set data for the Time vs. Molar_Amount graph *************** Index: (0, 0)
            curent_time = self.plot_data_state[0][-1]
            for i in range(self.reaction.n.shape[0]):
                self._plot_lines_amount[i].set_xdata(self.plot_data_state[0])
                self._plot_lines_amount[i].set_ydata(self.plot_data_mol[i])

                # set data for the Time vs. Molar_Concentration graph *************** Index: (0, 1)
                self._plot_lines_concentration[i].set_xdata(self.plot_data_state[0])
                self._plot_lines_concentration[i].set_ydata(self.plot_data_concentration[i])

            # reset each plot's x-limit because the latest action occurred at a greater time-value
            self._plot_axs[0, 0].set_xlim([0.0, curent_time])
            self._plot_axs[0, 1].set_xlim([0.0, curent_time])

            # set data for Time vs. Temp and Time vs. Volume graphs *************** Index: (0, 2)
            self._plot_line_1.set_xdata(self.plot_data_state[0])
            self._plot_line_1.set_ydata(self.plot_data_state[1])
            self._plot_line_2.set_xdata(self.plot_data_state[0])
            self._plot_line_2.set_ydata(self.plot_data_state[2])
            self._plot_axs[0, 2].set_xlim([0.0, curent_time]) # reset xlime since t extend

            # set data for Time vs. Pressure graph *************** Index: (1, 0)
            self._plot_line_3.set_xdata(self.plot_data_state[0])
            self._plot_line_3.set_ydata(self.plot_data_state[3])
            self._plot_axs[1, 0].set_xlim([0.0, curent_time]) # reset xlime since t extend

            # reset the Solid Spectra graph
            self._plot_axs[1, 1].cla()
            __ = self._plot_axs[1, 1].plot(
                wave, # wavelength space (ranging from wave_min to wave_max)
                absorb, # absorb spectra
            )[0]

            # obtain the new dash spectra data
            for __, item in enumerate(dash_spectra):
                self._plot_axs[1, 1].plot(wave, item, linestyle='dashed')

            # reset the spectra peak labelling
            for i in range(self.reaction.n.shape[0]):
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
