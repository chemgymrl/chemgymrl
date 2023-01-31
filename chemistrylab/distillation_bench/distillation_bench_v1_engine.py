'''
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

Distillation Demo
:title: distillation_bench_v1_engine.py
:author: Mitchell Shahen, Mark Baula
:history: 2020-07-23
'''

# pylint: disable=arguments-differ
# pylint: disable=protected-access
# pylint: disable=too-many-instance-attributes
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position

import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
from copy import deepcopy
import cmocean

sys.path.append("../../")
from chemistrylab.chem_algorithms import util, vessel
from chemistrylab.chem_algorithms.reward import DistillationReward
from chemistrylab.extract_algorithms import separate
from chemistrylab.distillations import distill_v0
from chemistrylab.reactions.reaction_base import _Reaction


class DistillationBenchEnv(gym.Env):
    '''
    Class object to create a Distillation environment.
    Parameters
    ---------------
    `boil_vessel` : `vessel` (default=`None`)
        A vessel object containing state variables, materials, solutes, and spectral data.
    `target_material` : `str` (default=`None`)
        The name of the required output material designated as reward.
    `n_steps` : `int` (default=`100`)
        The number of steps in an episode.
    Returns
    ---------------
    None
    Raises
    ---------------
    None
    '''

    def __init__(
            self,
            n_steps=100,
            boil_vessel=None,
            n_vessel_pixels=100,
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz",
            precipitation_file_identifier="precipitation",
            in_vessel_path=None,
            target_material=None,
            dQ=10.0,
            out_vessel_path=None
    ):
        '''
        Constructor class for the Distillation environment.
        '''

        # validate and assign the inputted parameters
        input_parameters = self._validate_parameters(
            n_steps=n_steps,
            boil_vessel=boil_vessel,
            n_vessel_pixels=n_vessel_pixels,
            reaction=reaction,
            reaction_file_identifier=reaction_file_identifier,
            precipitation_file_identifier=precipitation_file_identifier,
            in_vessel_path=in_vessel_path,
            target_material=target_material,
            dQ=dQ,
            out_vessel_path=out_vessel_path
        )

        self.reaction = input_parameters["reaction"](
            reaction_file_identifier=input_parameters["reaction_file_identifier"],
            target_material=target_material,
        )
        self.precipitation = input_parameters["reaction"](
            reaction_file_identifier=input_parameters["precipitation_file_identifier"],
            target_material=target_material,
        )

        # set the input parameters
        self.n_steps = input_parameters["n_steps"]
        self.boil_vessel = self._prepare_vessel(
            in_vessel_path=input_parameters["in_vessel_path"],
            default_vessel=deepcopy(input_parameters["boil_vessel"])
        )
        self.n_vessel_pixels = input_parameters["n_vessel_pixels"]
        self.target_material = input_parameters["target_material"]
        self.dQ = input_parameters["dQ"]
        self.out_vessel_path = input_parameters["out_vessel_path"]

        self.ni_steps = deepcopy(self.n_steps)

        # call the distillation class
        self.distillation = distill_v0.Distillation(
            boil_vessel=self.boil_vessel,
            target_material=self.target_material,
            dQ=self.dQ
        )

        # set up vessel and state variables
        self.vessels = None
        self.state = None

        # set up action spaces
        self.observation_space = self.distillation.get_observation_space(self.reaction.products)
        self.action_space = self.distillation.get_action_space()

        # set up the done boolean
        self.done = False

        # set up plotting parameters (TBD)
        self._first_render = True
        self._plot_fig = None
        self._plot_axs = None

        # the minimum purity of desired material in relation to total material that a finalized
        # distillation vessel must have to be qualified for use in other processes (the vessel
        # is saved if it exceeds this minimum purity threshold)
        self.min_purity_threshold = 0.5

    @staticmethod
    def _validate_parameters(n_steps=None, boil_vessel=None, n_vessel_pixels=100, reaction=None, reaction_file_identifier="", precipitation_file_identifier="", in_vessel_path=None, target_material=None, dQ=0.0, out_vessel_path=None):
        '''
        Checks and validates the input parameters submitted to the distillation bench.
        Parameters
        ---------------
        `n_steps` : `int` (default=`None`)
            The number of incremental steps to perform per action.
        `boil_vessel` : `vessel.Vessel` (default=`None`)
            The vessel object designated to be saved.
        `target_material` : `str` (default=`None`)
            The material in the boil vessel that is to be isolated.
        `dQ` : `float` (default=0.0):
            The maximal amount of heat to add to the boil vessel in one action.
        `out_vessel_path` : `str` (default=`None`)
            The directory path where the vessel object is to be saved.
        Returns
        ---------------
        `n_steps` : `int` (default=`100`)
            The number of incremental steps to perform per action.
        `boil_vessel` : `vessel.Vessel` (default=`None`)
            The vessel object designated to be saved.
        `target_material` : `str` (default=`None`)
            The material in the boil vessel that is to be isolated.
        `dQ` : `float` (default=0.0):
            The maximal amount of heat to add to the boil vessel in one action.
        `out_vessel_path` : `str` (default=`None`)
            The directory path where the vessel object is to be saved.
        Raises
        ---------------
        `TypeError`:
            Raised when the `boil_vessel` parameter is invalid or incompatible.
        '''

        # check the number of steps per action is an integer and non-negative
        if any([
                not isinstance(n_steps, int),
                n_steps <= 0
        ]):
            print("Invalid 'Number of Steps per Action' type. The default will be provided.")
            n_steps = 1

        # check that the boil vessel is a vessel object and is properly labelled
        if not isinstance(boil_vessel, vessel.Vessel):
            raise TypeError("The boil vessel provided is of an incorrect/incompatible type.")

        # ensure that the boil vessel is properly labelled
        boil_vessel.label = "boil_vessel"

        # ensure that the number of pixels is given as an integer
        if not isinstance(n_vessel_pixels, int):
            print("Invalid 'Number of Pixels' type. The default will be provided.")
            n_vessel_pixels = 100
        
        # ensure the reaction class is a class type object
        if not isinstance(reaction, type):
            raise TypeError("Invalid `Reaction Class` type. Unable to Proceed.")

        # ensure the reaction file identifier is a string a points to a legitimate file
        if not isinstance(reaction_file_identifier, str):
            raise TypeError("Invalid `Reaction File Parameter` type. Unable to Proceed.")

        # ensure the precipitation file identifier is a string a points to a legitimate file
        if not isinstance(precipitation_file_identifier, str):
            raise TypeError("Invalid `Precipitation File Parameter` type. Unable to Proceed.")

        # ensure the reaction file identifier is a string a points to a legitimate file
        if not isinstance(in_vessel_path, str) and not (in_vessel_path is None):
            raise TypeError("Invalid `Reaction File Parameter` type. Unable to Proceed.")

        # ensure that the target material is given as a string
        if not isinstance(target_material, str):
            print("Invalid 'Target Material' type. The default will be provided.")
            target_material = ""

        # ensure that the maximal heat increment parameter is a float or an int value
        if all([
                not isinstance(dQ, float),
                not isinstance(dQ, int)
        ]):
            print("Invalid 'Maximal Heat Increment' type. The default will be provided.")
            dQ = 0.0
        else:
            # ensure the heat increment is a float value
            dQ = float(dQ)

        # ensure that the output vessel path parameter is provided as a string
        if not isinstance(out_vessel_path, str):
            print("Invalid 'Output Vessel Path' type. The default will be provided.")
            out_vessel_path = os.getcwd()

        # validate that the output vessel path given exists and is reachable
        if not os.path.isdir(out_vessel_path):
            print(
                "The given output vessel path parameter is invalid. "
                "The default parameter is given instead."
            )
            out_vessel_path = os.getcwd()

        # collect the input parameters into a labelled list
        input_parameters = {
            "n_steps" : n_steps,
            "boil_vessel" : boil_vessel,
            "n_vessel_pixels" : n_vessel_pixels,
            "reaction" : reaction,
            "reaction_file_identifier" : reaction_file_identifier,
            "precipitation_file_identifier" : precipitation_file_identifier, 
            "in_vessel_path" : in_vessel_path,
            "target_material" : target_material,
            "dQ" : dQ,
            "out_vessel_path" : out_vessel_path
        }

        return input_parameters

    def _prepare_vessel(self, in_vessel_path=None, default_vessel=None):
        """
        Method to prepare the initial vessel.

        Parameters
        ---------------
        `in_vessel_path` : `str` (default=`None`)
            A string indicating the path to a vessel intended to be loaded into the reaction bench environment.
        `materials` : `list` (default=`None`)
            A list of dictionaries including initial material names and amounts.
        `solvents` : `list` (default=`None`)
            A list of dictionaries including initial solvent names and amounts.

        Returns
        ---------------
        `vessels` : `vessel.Vessel`
            A vessel object containing materials and solvents to be used in the reaction bench.

        Raises
        ---------------
        None
        """
        # initialize vessels by providing a empty default vessel or loading an existing saved vessel
        if in_vessel_path is None:
            vessels = deepcopy(default_vessel)

        else:
            with open(in_vessel_path, 'rb') as handle:
                vessels = pickle.load(handle)

        return vessels

    def update_vessel(self, vessel):
        self.boil_vessel = vessel
        self.distillation = distill_v0.Distillation(
            boil_vessel=self.boil_vessel,
            target_material=self.target_material,
            dQ=1e+5,
        )

    def _save_vessel(self, distillation_vessel=None, vessel_rootname=""):
        '''
        Method to save a vessel as a pickle file.

        Parameters
        ---------------
        `distillation_vessel` : `vessel.Vessel` (default=`None`)
            The vessel object designated to be saved.
        `vessel_rootname` : `str` (default="")
            The intended name/identifier of the pickle file to which the vessel is being saved.
        Returns
        ---------------
        None
        Raises
        ---------------
        None
        '''

        # specify a file to save the distillation vessel
        file_directory = self.out_vessel_path
        filename = "{}.pickle".format(vessel_rootname)
        open_file = os.path.join(file_directory, filename)

        # delete any existing vessel files to ensure the vessel is saved as intended
        if os.path.exists(open_file):
            os.remove(open_file)

        # open the intended vessel file and save the vessel as a pickle file
        with open(open_file, 'wb') as vessel_file:
            pickle.dump(distillation_vessel, vessel_file)

    def _update_state(self):
        '''
        Method to update the state variable.

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

        # define the state variable using information from each available vessel
        # state[i] = array describing the ith vessel
        # state[i][:n_vessel_pixels] = vessel layers
        # state[i][n_vessel_pixels] = temperature
        # state[i][n_vessel_pixels+1] = volume
        # state[i][n_vessel_pixels+2] = pressure
        
        # set up the base state variable
        self.state = np.zeros((len(self.vessels), self.distillation.n_vessel_pixels+3+len(self.reaction.products)), dtype=np.float32)

        self.state[:, :self.distillation.n_vessel_pixels] = util.generate_layers_obs(
            self.vessels,
            max_n_vessel=self.distillation.n_total_vessels,
            n_vessel_pixels=self.distillation.n_vessel_pixels
        )

        # iterate through each available vessel
        for i, vessel in enumerate(self.vessels):
            # set up the temperature
            Tmin = vessel.get_Tmin()
            Tmax = vessel.get_Tmax()
            temp = vessel.get_temperature()
            normalized_temp = (temp - Tmin) / (Tmax - Tmin)
            self.state[i, self.distillation.n_vessel_pixels] = normalized_temp

            # set up the volume
            Vmin = vessel.get_min_volume()
            Vmax = vessel.get_max_volume()
            volume = vessel.get_volume()
            normalized_volume = (volume - Vmin) / (Vmax - Vmin)
            self.state[i, self.distillation.n_vessel_pixels+1] = normalized_volume

            # set up the pressure
            Pmax = vessel.get_pmax()
            total_pressure = vessel.get_pressure()
            normalized_pressure = total_pressure / Pmax
            self.state[i, self.distillation.n_vessel_pixels+2] = normalized_pressure

        targ_ind = self.reaction.products.index(self.target_material)
        self.state[:, self.distillation.n_vessel_pixels+3+targ_ind] += 1

    def reset(self):
        '''
        Initialize the environment
        Parameters
        ---------------
        None
        Returns
        ---------------
        `state` : `np.array`
            A numpy array containing state variables, material concentrations, and spectral data.
        Raises
        ---------------
        None
        '''

        # reset the done and plotting parameters to initial values
        self.done = False
        self.n_steps = deepcopy(self.ni_steps)
        self._first_render = True

        # acquire the initial vessels from the distillation class
        self.vessels = self.distillation.reset(
            init_boil_vessel=self.boil_vessel
        )
        self.vessels[0] = self.precipitation.reset(vessels=self.vessels[0])

        # update the state variables
        self._update_state()

        return self.state

    def step(self, action):
        '''
        Update the environment by performing an action.
        Parameters
        ---------------
        `action` : `list`
            A list of two numbers indicating the index of the action to
            perform and a multiplier to use in completing the action.
        Returns
        ---------------
        `state` : `np.array`
            A numpy array containing state variables, material concentrations, and spectral data.
        `reward` : `float`
            The amount of reward generated in perfoming the action.
        `done` : `bool`
            A boolean indicting if all the steps in an episode have been completed.
        `params` : `dict`
            A dictionary containing additional parameters or information that may be useful.
        Raises
        ---------------
        None
        '''

        # perform the action and update the vessels, interim reward, and done variables
        self.vessels, reward, self.done = self.distillation.perform_action(
            vessels=self.vessels,
            action=action
        )

        if len(self.vessels[0]._material_dict) > 0:
            prep_action = np.zeros(len(self.precipitation.reactants)+2)
            prep_action[0:2] += 0.5
            self.vessels[0] = self.precipitation.perform_action(
                vessels=self.vessels[0],
                action=prep_action,
                n_steps=50,
                t=0
            )

        # update the state variable
        self._update_state()

        # update the number of steps left to complete
        self.n_steps -= 1
        if any([self.n_steps == 0, self.done]):
            self.done = True

            # initialize the distillation reward class
            d_reward = DistillationReward(
                vessels=self.vessels,
                desired_material=self.target_material
            )

            # after the last step, calculate the final reward
            reward = d_reward.calc_reward()

            # obtain a list of the vessels that contain the desired
            # material and obey the minimum purity requirements
            valid_vessels = d_reward.validate_vessels(
                purity_threshold=self.min_purity_threshold
            )

            # save any finalized, validated vessels for future use or reference
            for i, vessel in enumerate(valid_vessels):
                self._save_vessel(
                    distillation_vessel=vessel,
                    vessel_rootname="distillation_vessel_{}".format(i)
                )

        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        '''
        Select a render mode to display pertinent information.

        Parameters
        ---------------
        `model` : `str` (default=`human`)
            The name of the render mode to use.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        if mode == 'human':
            self.human_render()
        elif mode == 'full':
            self.full_render()

    def human_render(self, mode='plot'):
        '''
        Render the pertinent information in a minimal style for the user to visualize and process.

        Parameters
        ---------------
        `mode` : `str` (default=`plot`)
            The type of rendering to use.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # create a list containing an array for each vessel in `self.vessels`
        position_separate = []
        for vessel_obj in self.vessels:
            position, __ = vessel_obj.get_position_and_variance()

            # append an array of shape: (# of layers in vessel, # of points in the spectral plot)
            position_separate.append(
                np.zeros((len(position), separate.x.shape[0]), dtype=np.float32)
            )

        # iterate through each array and populate them with gaussian data for each material layer
        for i, arr in enumerate(position_separate):
            position, var = self.vessels[i].get_position_and_variance(dict_or_list='dict')
            t = -1.0 * np.log(var * np.sqrt(2.0 * np.pi))
            j = 0

            for layer in position:
                # Loop over each layer
                mat_vol = self.vessels[i].get_material_volume(layer)

                for k in range(arr.shape[1]):
                    # Calculate the value of the Gaussian for that phase and x position
                    exponential = np.exp(
                        -1.0 * (((separate.x[k] - position[layer]) / (2.0 * var)) ** 2) + t
                    )

                    # populate the array
                    arr[j, k] = mat_vol * exponential
                j += 1

        Ls = np.reshape(
            np.array(
                self.state[:, :self.n_vessel_pixels]
            ),
            (
                self.distillation.n_total_vessels,
                self.distillation.n_vessel_pixels,
                1
            )
        )

        num_list = [self.state[i, self.distillation.n_vessel_pixels],
                    self.state[i, self.distillation.n_vessel_pixels+1],
                    self.state[i, self.distillation.n_vessel_pixels+2]
                    ]

        label_list = ['T', 'V', 'P']

        # set parameters and plotting for the first rendering
        if self._first_render:
            # close all existing plots and enable interactive mode
            plt.close('all')
            plt.ion()

            # define a set of three subplots
            self._plot_fig, self._plot_axs = plt.subplots(
                2,
                self.distillation.n_total_vessels,
                figsize=(12, 6)
            )

            # extract each array of plotting data, determine the plot colour, and add the plot
            for i, arr in enumerate(position_separate):
                # define the mixing visualization graphic rendered beside the current subplot
                __ = self._plot_axs[0, i].pcolormesh(
                    Ls[i],
                    vmin=0,
                    vmax=1,
                    cmap=cmocean.cm.delta
                )

                # set plotting parameters for the mixing graphic
                self._plot_axs[0, i].set_xticks([])
                self._plot_axs[0, i].set_ylabel('Height')

                # define visualization of vessel observation properties
                __ = self._plot_axs[1, i].bar(
                range(len(num_list)),
                num_list,
                tick_label=label_list,
                )[0]
                self._plot_axs[1, i].set_ylim([0, 1])

                # draw the canvas and render the subplot
                self._plot_fig.canvas.draw()
                plt.show()

                self._first_render = False

        # if the plot has already been rendered, update the plot
        else:
            # iterate through each array
            for i, arr in enumerate(position_separate):                
                # define the layer mixture graphic beside the current subplot
                __ = self._plot_axs[0, i].pcolormesh(
                    Ls[i],
                    vmin=0,
                    vmax=1,
                    cmap=cmocean.cm.delta
                )

                self._plot_axs[1, i].cla()
                # define visualization of vessel observation properties
                __ = self._plot_axs[1, i].bar(
                    range(len(num_list)),
                    num_list,
                    tick_label=label_list,
                )[0]
                self._plot_axs[1, i].set_ylim([0, 1])

                # draw the subplot on the existing canvas
                self._plot_fig.canvas.draw()
                # plt.show() # adding this causes the plot to be unresponsive when updating

                self._first_render = False

    def full_render(self, mode='plot'):
        '''
        Render the pertinent information in a minimal style for the user to visualize and process.

        Parameters
        ---------------
        `mode` : `str` (default=`plot`)
            The type of rendering to use.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # create a list containing an array for each vessel in `self.vessels`
        position_separate = []
        for vessel_obj in self.vessels:
            position, __ = vessel_obj.get_position_and_variance()

            # append an array of shape: (# of layers in vessel, # of points in the spectral plot)
            position_separate.append(
                np.zeros((len(position), separate.x.shape[0]), dtype=np.float32)
            )

        # iterate through each array and populate them with gaussian data for each material layer
        for i, arr in enumerate(position_separate):
            position, var = self.vessels[i].get_position_and_variance(dict_or_list='dict')
            t = -1.0 * np.log(var * np.sqrt(2.0 * np.pi))
            j = 0

            for layer in position:
                # Loop over each layer
                mat_vol = self.vessels[i].get_material_volume(layer)

                for k in range(arr.shape[1]):
                    # Calculate the value of the Gaussian for that phase and x position
                    exponential = np.exp(
                        -1.0 * (((separate.x[k] - position[layer]) / (2.0 * var)) ** 2) + t
                    )

                    # populate the array
                    arr[j, k] = mat_vol * exponential
                j += 1

        Ls = np.reshape(
            np.array(
                self.state[:, :self.n_vessel_pixels]
            ),
            (
                self.distillation.n_total_vessels,
                self.distillation.n_vessel_pixels,
                1
            )
        )

        num_list = [self.state[i, self.distillation.n_vessel_pixels],
                    self.state[i, self.distillation.n_vessel_pixels+1],
                    self.state[i, self.distillation.n_vessel_pixels+2]
                    ]

        label_list = ['T', 'V', 'P']

        # set parameters and plotting for the first rendering
        if self._first_render:
            # close all existing plots and enable interactive mode
            plt.close('all')
            plt.ion()

            # define a set of three subplots
            self._plot_fig, self._plot_axs = plt.subplots(
                3,
                self.distillation.n_total_vessels,
                figsize=(12, 6)
            )

            # extract each array of plotting data, determine the plot colour, and add the plot
            for i, arr in enumerate(position_separate):
                # define the mixing visualization graphic rendered beside the current subplot
                __ = self._plot_axs[0, i].pcolormesh(
                    Ls[i],
                    vmin=0,
                    vmax=1,
                    cmap=cmocean.cm.delta
                )

                # set plotting parameters for the mixing graphic
                self._plot_axs[0, i].set_xticks([])
                self._plot_axs[0, i].set_ylabel('Height')

                # define visualization of vessel observation properties
                __ = self._plot_axs[1, i].bar(
                range(len(num_list)),
                num_list,
                tick_label=label_list,
                )[0]
                self._plot_axs[1, i].set_ylim([0, 1])

                count = 0
                material_dict = self.vessels[i].get_material_dict()
                for mat in material_dict.keys():
                    count += 1
                    self._plot_axs[2, i].bar(count, material_dict[mat][1])

                self._plot_axs[2, i].set_xticks(range(1, count+1))
                self._plot_axs[2, i].set_xticklabels(material_dict.keys())

                # draw the canvas and render the subplot
                self._plot_fig.canvas.draw()
                plt.show()

                self._first_render = False

        # if the plot has already been rendered, update the plot
        else:
            # iterate through each array
            for i, arr in enumerate(position_separate):                
                # define the layer mixture graphic beside the current subplot
                __ = self._plot_axs[0, i].pcolormesh(
                    Ls[i],
                    vmin=0,
                    vmax=1,
                    cmap=cmocean.cm.delta
                )

                self._plot_axs[1, i].cla()
                # define visualization of vessel observation properties
                __ = self._plot_axs[1, i].bar(
                    range(len(num_list)),
                    num_list,
                    tick_label=label_list,
                )[0]
                self._plot_axs[1, i].set_ylim([0, 1])

                self._plot_axs[2, i].cla()
                count = 0
                material_dict = self.vessels[i].get_material_dict()
                for mat in material_dict.keys():
                    count += 1
                    self._plot_axs[2, i].bar(count, material_dict[mat][1])

                self._plot_axs[2, i].set_xticks(range(1, count+1))
                self._plot_axs[2, i].set_xticklabels(material_dict.keys())

                # draw the subplot on the existing canvas
                self._plot_fig.canvas.draw()
                # plt.show() # adding this causes the plot to be unresponsive when updating

                self._first_render = False
