'''
Reaction Bench Engine Class

:title: reaction_bench_v0_engine

:author: Chris Beeler and Mitchell Shahen

:history: 21-05-2020
'''

# pylint: disable=attribute-defined-outside-init
# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-lines
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
import datetime as dt

sys.path.append("../../") # to access chemistrylab
sys.path.append("../reactions/") # to access all reactions
from chemistrylab.chem_algorithms.reward import ReactionReward
from chemistrylab.chem_algorithms import vessel, util
from chemistrylab.reactions.get_reactions import convert_to_class
from chemistrylab.reactions.reaction_base import _Reaction

R = 0.008314462618 # Gas constant (kPa * m**3 * mol**-1 * K**-1)
wave_max = 800
wave_min = 200

class ReactionBenchEnv(gym.Env):
    '''
    Class to define elements of an engine to represent a reaction.
    '''

    def __init__(
            self,
            reaction=_Reaction,
            reaction_file_identifier="",
            in_vessel_path=None,
            out_vessel_path=None,
            materials=None,
            solutes=None,
            n_steps=50,
            dt=0.01,
            overlap=False
    ):
        '''
        Constructor class method to pass thermodynamic variables to class methods.

        Parameters
        ---------------
        `reaction` : `class`
            A reaction class containing various methods that
            properly define a reaction to be carried out.
        `in_vessel_path` : `str` (default=`None`)
            A string indicating the path to a vessel intended to be loaded into this module.
        `out_vessel_path` : `str` (default=`None`)
            A string indicating the path to a directory where the final output vessel is saved.
        `materials` : `list` (default=`None`)
            A list of dictionaries including initial material names, classes, and amounts.
        `solutes` : `list` (default=`None`)
            A list of dictionaries including initial solute names, classes, and amounts.
        `n_steps` : `int` (default=`50`)
            The number of time steps to be taken during each action.
        `dt` : `float` (default=`0.01`)
            The amount of time taken in each time step.
        `overlap` : `boolean` (default=`False`)
            Indicates if the spectral plots show overlapping signatures.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # assign an identifier to this bench
        self.name = "react_bench"

        # validate the parameters inputted to the reaction bench engine
        input_parameters = self._validate_parameters(
            reaction=reaction,
            reaction_file_identifier=reaction_file_identifier,
            in_vessel_path=in_vessel_path,
            out_vessel_path=out_vessel_path,
            materials=materials,
            solutes=solutes,
            n_steps=n_steps,
            dt=dt,
            overlap=overlap
        )

        # set the input vessel path
        self.in_vessel_path = input_parameters["in_vessel_path"]

        # set the output vessel path
        self.out_vessel_path = input_parameters["out_vessel_path"]

        self.n_steps = input_parameters["n_steps"] # Time steps per action
        self.dt = input_parameters["dt"] # Time step of reaction (s)

        # initialize the reaction
        self.reaction = input_parameters["reaction"](
            reaction_file_identifier=input_parameters["reaction_file_identifier"],
            overlap=input_parameters["overlap"]
        )

        # Maximum time (s) (20 is the registered max_episode_steps)
        self.tmax = self.n_steps * self.dt * 20
        self.t = 0

        # initialize a step counter
        self.step_num = 1

        # initialize vessels by providing a empty default vessel or loading an existing saved vessel
        self.n_init = np.zeros(self.reaction.nmax.shape[0], dtype=np.float32)
        if self.in_vessel_path is None:
            # prepare the materials that have been provided
            material_names = []
            material_amounts = []
            for material_params in input_parameters["materials"]:
                material_names.append(material_params["Material"])
                material_amounts.append(material_params["Initial"])
            material_classes = convert_to_class(materials=material_names)
            material_dict = {}
            for i, material in enumerate(material_names):
                material_dict[material] = [material_classes[i], material_amounts[i], 'mol']

            # prepare the solutes that have been provided
            solute_names = []
            solute_amounts = []
            for solute_params in input_parameters["solutes"]:
                solute_names.append(solute_params["Solute"])
                solute_amounts.append(solute_params["Initial"])
            solute_classes = convert_to_class(materials=solute_names)
            solute_dict = {}
            for i, solute in enumerate(solute_names):
                solute_dict[solute] = [solute_classes[i], solute_amounts[i], 'mol']

            # add the materials and solutes to an empty vessel
            self.vessels = vessel.Vessel(
                'default',
                materials=material_dict,
                solutes=solute_dict,
                default_dt=self.dt
            )
        else:
            with open(self.in_vessel_path, 'rb') as handle:
                v = pickle.load(handle)
                self.vessels = v

        # set up a state variable
        self.state = None

        # reset the inputted reaction before performing any steps
        self.reaction.reset(vessels=self.vessels)

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
        # between 0.0 and 1.0) for a wide range of wavelengths;
        # only needed for specifying the observation space so only the array size is required
        absorb = self.reaction.get_spectra(self.vessels.get_volume())

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

    @staticmethod
    def _validate_parameters(
            reaction=None,
            reaction_file_identifier="",
            in_vessel_path="",
            out_vessel_path="",
            materials=None,
            solutes=None,
            n_steps=0,
            dt=0.0,
            overlap=False
    ):
        '''
        Method to validate the parameters inputted to the reaction class.

        Parameters
        ---------------
        `reaction` : `class`
            A reaction class containing various methods that
            properly define a reaction to be carried out.
        `in_vessel_path` : `str` (default=`""`)
            A string indicating the path to a vessel intended to be loaded into this module.
        `out_vessel_path` : `str` (default=`""`)
            A string indicating the path to a directory where the final output vessel is saved.
        `materials` : `list` (default=`None`)
            A list of dictionaries including initial material names, classes, and amounts.
        `solutes` : `list` (default=`None`)
            A list of dictionaries including initial solute names, classes, and amounts.
        `n_steps` : `int` (default=`0`)
            The number of time steps to be taken during each action.
        `dt` : `float` (default=`0.0`)
            The amount of time taken in each time step.
        `overlap` : `boolean` (default=`False`)
            Indicates if the spectral plots show overlapping signatures.

        Returns
        ---------------
        `input_parameters` : `dict`
            A dictionary containing all the validated input parameters.

        Raises
        ---------------
        `TypeError`:
            Raised when the `reaction` parameter is invalid.
        '''

        ## ----- ## CHECK PARAMETER TYPES ## ----- ##

        # ensure the reaction class is a class type object
        if not isinstance(reaction, type):
            raise TypeError("Invalid `Reaction Class` type. Unable to Proceed.")

        # ensure the reaction file identifier is a string a points to a legitimate file
        if not isinstance(reaction_file_identifier, str):
            raise TypeError("Invalid `Reaction File Parameter` type. Unable to Proceed.")

        # ensure the input vessel parameter points to a legitimate location or is `None`
        if in_vessel_path is not None:
            if not isinstance(in_vessel_path, str):
                print("The provided input vessel path is invalid. The default will be provided.")
                in_vessel_path = None
            elif os.path.isfile(in_vessel_path):
                if in_vessel_path.split(".")[-1] == "pickle":
                    pass
                else:
                    print("The provided input vessel path is invalid. The default will be provided.")
            else:
                print("The provided input vessel path is invalid. The default will be provided.")
                in_vessel_path = None

        # ensure the output vessel parameter points to a legitimate directory
        if not isinstance(out_vessel_path, str):
            print("The provided output vessel path is invalid. The default will be provided.")
            out_vessel_path = os.getcwd()
        elif os.path.isdir(out_vessel_path):
            pass
        else:
            print("The provided output vessel path is invalid. The default will be provided.")
            out_vessel_path = os.getcwd()

        # ensure the materials parameter is a non-empty list
        if not isinstance(materials, list):
            print("Invalid `materials` type. The default will be provided.")
            materials = []
        if len(materials) == 0:
            print("Error: The materials list contains no elements.")

        # ensure the solutes parameter is a non-empty list
        if not isinstance(solutes, list):
            print("Invalid `materials` type. The default will be provided.")
            solutes = []
        if len(solutes) == 0:
            print("Error: The materials list contains no elements.")

        # ensure the n_steps parameter is a non-zero, non-negative integer
        if any([
                not isinstance(n_steps, int),
                n_steps <= 0
        ]):
            print("Invalid 'Number of Steps per Action' type. The default will be provided.")
            n_steps = 1

        # the default timestep parameter must be a non-zero, non-negative floating point value
        if any([
                not isinstance(dt, float),
                dt <= 0.0
        ]):
            print("Invalid 'Default Timestep' type. The default will be provided.")
            dt = 1.0

        # the overlap parameter must be a boolean
        if not isinstance(overlap, bool):
            print("Invalid 'Overlap Spectra' type. The default will be provided.")
            overlap = False

        # collect all the input parameters in a labelled dictionary
        input_parameters = {
            "reaction" : reaction,
            "reaction_file_identifier" : reaction_file_identifier,
            "in_vessel_path" : in_vessel_path,
            "out_vessel_path" : out_vessel_path,
            "materials" : materials,
            "solutes" : solutes,
            "n_steps" : n_steps,
            "dt" : dt,
            "overlap" : overlap
        }

        return input_parameters

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

        T = self.vessels.get_temperature()
        V = self.vessels.get_volume()

        absorb = self.reaction.get_spectra(V)

        # create an array to contain all state variables
        state = np.zeros(
            4 +  # time T V P
            len(self.reaction.reactants) +  # reactants
            absorb.shape[0],  # spectra
            dtype=np.float32
        )

        # populate the state array with updated state variables
        # state[0] = time
        state[0] = self.t / self.tmax

        # state[1] = temperature
        Tmin = self.vessels.get_Tmin()
        Tmax = self.vessels.get_Tmax()
        state[1] = (T - Tmin) / (Tmax - Tmin)

        # state[2] = volume
        Vmin = self.vessels.get_min_volume()
        Vmax = self.vessels.get_max_volume()
        state[2] = (V - Vmin) / (Vmax - Vmin)

        # state[3] = pressure
        total_pressure = self.vessels.get_pressure()
        Pmax = self.vessels.get_pmax()
        state[3] = total_pressure / Pmax

        # remaining state variables pertain to reactant amounts and spectra
        for i in range(self.reaction.initial_in_hand.shape[0]):
            state[i + 4] = self.reaction.n[i] / self.reaction.nmax[i]
        for i in range(absorb.shape[0]):
            state[i + self.reaction.initial_in_hand.shape[0] + 4] = absorb[i]

        self.state = state

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

        # reinitialize the reaction class
        self.vessels = self.reaction.reset(vessels=self.vessels)
        
        Ti = self.vessels.get_temperature()
        Tmin = self.vessels.get_Tmin()
        Tmax = self.vessels.get_Tmax()
        Vi = self.vessels.get_volume()
        Vmin = self.vessels.get_min_volume()
        Vmax = self.vessels.get_max_volume()
        total_pressure = self.vessels.get_pressure()
        Pmax = self.vessels.get_pmax()

        # populate the state with the above variables
        self.plot_data_state = [
            [0.0], # time
            [(Ti - Tmin) / (Tmax - Tmin)], # normalized temperature
            [(Vi - Vmin) / (Vmax - Vmin)], # normalized volume
            [total_pressure / Pmax] # normalized pressure
        ]
        self.plot_data_mol = []
        self.plot_data_concentration = []

        # [0] is time, [1] is Tempurature, [2:] is each species
        C = self.reaction.get_concentration(Vi)
        for i in range(self.reaction.nmax.shape[0]):
            self.plot_data_mol.append([self.reaction.n[i]])
            self.plot_data_concentration.append([C[i]])

        # update the state variables with the initial values
        self._update_state()

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

        # the reward for this step is set to 0 initially
        reward = 0.0

        # pass the action and vessel to the reaction base class's perform action function
        self.vessels = self.reaction.perform_action(action, self.vessels, self.n_steps)

        # call some function to get plotting data

        self._update_state()

        # calculate the reward for this step
        reward = ReactionReward(
            vessel=self.vessels,
            desired_material=self.reaction.desired
        ).calc_reward()
        # ---------- ADD ---------- #
        # add option to save or print intermediary vessel

        # save the vessel when the final step is complete
        if any([self.done, self.step_num == 20]):
            self.vessels.save_vessel('reaction_vessel')

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

        reactants = []
        for reactant in self.reaction.reactants:
            name = reactant.replace("[", "").replace("]", "")
            short_name = name[0:5]
            reactants.append(short_name)

        label_list = ['t', 'T', 'V', 'P'] + reactants

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

        reactants = []
        for reactant in self.reaction.reactants:
            name = reactant.replace("[", "").replace("]", "")
            short_name = name[0:4]
            reactants.append(short_name)

        label_list = ['t', 'T', 'V', 'P'] + reactants

        # get the spectral data peak and dashed spectral lines
        peak = self.reaction.get_spectra_peak(self.vessels.get_volume())
        dash_spectra = self.reaction.get_dash_line_spectra(self.vessels.get_volume())

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
                        label=self.reaction.materials[i] # names of species C[i]
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
                        label=self.reaction.materials[i] # names of species C[i]
                    )[0]
                )
            self._plot_axs[0, 1].set_xlim([0.0, self.vessels.get_defaultdt() * self.n_steps])
            self._plot_axs[0, 1].set_ylim([
                0.0, np.max(self.reaction.nmax)/(self.vessels.get_min_volume())
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
