'''
Distillation Demo

:title: distillation_bench_v1_engine.py

:author: Mitchell Shahen

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

sys.path.append("../../")
from chemistrylab.chem_algorithms import util, vessel
from chemistrylab.chem_algorithms.reward import DistillationReward
from chemistrylab.distillations import distill_v0

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
            target_material=None,
            out_vessel_path=None
    ):
        '''
        Constructor class for the Distillation environment.
        '''

        # validate and assign the inputted parameters
        input_parameters = self._validate_parameters(
            n_steps=n_steps,
            boil_vessel=boil_vessel,
            target_material=target_material,
            out_vessel_path=out_vessel_path
        )

        # set the input parameters
        self.n_steps = input_parameters["n_steps"]
        self.boil_vessel = input_parameters["boil_vessel"]
        self.target_material = input_parameters["target_material"]
        self.out_vessel_path = input_parameters["out_vessel_path"]

        # call the distillation class
        self.distillation = distill_v0.Distillation(
            boil_vessel=self.boil_vessel,
            target_material=self.target_material,
            dQ=1e+5
        )

        # set up vessel and state variables
        self.vessels = None
        self.state = None

        # set up action spaces
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
    def _validate_parameters(n_steps=None, boil_vessel=None, target_material=None, out_vessel_path=None):
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

        # ensure that the target material is given as a string
        if not isinstance(target_material, str):
            print("Invalid 'Target Material' type. The default will be provided.")
            target_material = ""

        # check that the target material is present in the boil vessel
        if not target_material in boil_vessel._material_dict.keys():
            print(
                "The target material, {}, is not present in the boil vessel's material dictionary.".format(
                    target_material
                )
            )
            target_material = ""

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
            "target_material" : target_material,
            "out_vessel_path" : out_vessel_path
        }

        return input_parameters

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
        # state[i][0] = vessel name
        # state[i][1] = temperature
        # state[i][2] = volume
        # state[i][3] = pressure
        # state[i][4:] = array containing the materials in the ith vessel

        # set up the base state variable
        base_state = [[] for __ in enumerate(self.vessels)]

        # iterate through each available vessel
        for i, vessel in enumerate(self.vessels):
            # get the vessel name
            vessel_name = vessel.label
            base_state[i].append(vessel_name)

            # set up the temperature
            Tmin = vessel.get_Tmin()
            Tmax = vessel.get_Tmax()
            temp = vessel.get_temperature()
            normalized_temp = (temp - Tmin) / (Tmax - Tmin)
            base_state[i].append(normalized_temp)

            # set up the volume
            Vmin = vessel.get_min_volume()
            Vmax = vessel.get_max_volume()
            volume = vessel.get_volume()
            normalized_volume = (volume - Vmin) / (Vmax - Vmin)
            base_state[i].append(normalized_volume)

            # set up the pressure
            Pmax = vessel.get_pmax()
            total_pressure = vessel.get_pressure()
            normalized_pressure = total_pressure / Pmax
            base_state[i].append(normalized_pressure)

            # set up each set of material properties as separate packages (as lists)
            # each list has three elements: [material_name, material_class, material_amount]
            material_packages = []
            for material_name, [material_class, material_amount] in vessel._material_dict.items():
                material_package = [material_name, material_class, material_amount]
                material_packages.append(material_package)
            for material_package in material_packages:
                base_state[i].append(material_package)

        # convert the base state to a numpy array to be accessible for all methods
        self.state = np.array(base_state)

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
        self._first_render = True

        # acquire the initial vessels from the distillation class
        vessels = self.distillation.reset(
            boil_vessel=self.boil_vessel
        )

        # re-define the existing vessels
        self.vessels = vessels

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

        print(self.state)
        __ = input("Wait...")

        # obtain the necessary variables
        vessels = self.vessels
        done = self.done

        # perform the action and update the vessels, reward, and done variables
        vessels, reward, done = self.distillation.perform_action(
            vessels=vessels,
            action=action
        )

        # redefine the global variables
        self.vessels = vessels
        self.done = done

        # update the state variable
        self._update_state()

        # update the number of steps left to complete
        self.n_steps -= 1
        if any([self.n_steps == 0, self.done]):
            self.done = True

            # after the last step, calculate the final reward
            reward = DistillationReward(
                vessels=self.vessels,
                desired_material=self.target_material
            ).calc_reward()

            # obtain a list of the vessels that contain the desired
            # material and obey the minimum purity requirements
            valid_vessels = DistillationReward(
                vessels=self.vessels,
                desired_material=self.target_material
            ).validate_vessels(
                purity_threshold=self.min_purity_threshold
            )

            # save any finalized, validated vessels for future use or reference
            for i, vessel in enumerate(valid_vessels):
                self._save_vessel(
                    distillation_vessel=vessel,
                    vessel_rootname="distillation_vessel_{}".format(i)
                )

        return self.state, reward, self.done, {}

    def render(self, model):
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

        if model == 'human':
            self.human_render()

    def human_render(self):
        '''
        Method to render a series of graphs to illustrate operations on vessels.
        '''

        # obtain all vessels
        vessels = self.vessels

        # separate the beakers from the boil vessel
        beakers = []
        for vessel_obj in vessels:
            if "beaker" in vessel_obj.label:
                beakers.append(vessel_obj)

        # there is only one boil vessel
        boil_vessel = [vessel_obj for vessel_obj in vessels if vessel_obj not in beakers][0]

        if self._first_render:
            plt.close('all')
            plt.ion()

            num_columns = max(len(beakers), 2)
            self._plot_fig, self._plot_axs = plt.subplots(
                2, # number of rows
                num_columns, # number of columns
                figsize=(18, 9)
            )

            # create a subplot for the boil vessel at position (0, 0)
            self._plot_bar1 = self._plot_axs[0, 0].bar(
                [name[0:5] for name in boil_vessel._material_dict.keys()],
                [value[1] for value in boil_vessel._material_dict.values()],
                color='red'
            )
            self._plot_axs[0, 0].set_title("Boil Vessel")
            self._plot_axs[0, 0].set_ylabel("Molar Amount")

            # create a subplot for the vessel temperatures at position (0, 1)
            self._plot_bar2 = self._plot_axs[0, 1].bar(
                [vessel_obj.label for vessel_obj in vessels],
                [vessel_obj.temperature for vessel_obj in vessels],
                color='orange'
            )
            self._plot_axs[0, 1].set_title("Vessel Temperatures")
            self._plot_axs[0, 1].set_ylabel('Temperature (in K)')

            # create subplots in the second row for each beaker
            self._beaker_plots = []
            for i, beaker_obj in enumerate(beakers):
                material_names = [name[0:5] for name in beaker_obj._material_dict.keys()]
                material_amounts = [value[1] for value in beaker_obj._material_dict.values()]

                beaker_plot = self._plot_axs[1, i].bar(material_names, material_amounts, color='blue')
                self._plot_axs[1, i].set_title(beaker_obj.label)
                self._plot_axs[1, i].set_ylabel("Molar Amount")

                self._beaker_plots.append(beaker_plot)

            # draw the graph and show it
            self._plot_fig.canvas.draw()
            plt.show()

            self._first_render = False

        # if the plot has already been rendered, simply add the new data to it
        else:
            # reset the boil vessel plot with new vessel data
            __ = self._plot_axs[0, 0].bar(
                [name[0:5] for name in boil_vessel._material_dict.keys()],
                [value[1] for value in boil_vessel._material_dict.values()],
                color='red'
            )
            self._plot_axs[0, 0].set_title("Boil Vessel")
            self._plot_axs[0, 0].set_ylabel("Molar Amount")

            # reset the vessel temperature plot with new data
            __ = self._plot_axs[0, 1].bar(
                [vessel_obj.label for vessel_obj in vessels],
                [vessel_obj.temperature for vessel_obj in vessels],
                color='orange'
            )
            self._plot_axs[0, 1].set_title("Vessel Temperatures")
            self._plot_axs[0, 1].set_ylabel('Temperature (in K)')

            # reset the data in each beaker plot
            for i, beaker_plot in enumerate(self._beaker_plots):
                material_names = [name[0:5] for name in beakers[i]._material_dict.keys()]
                material_amounts = [value[1] for value in beakers[i]._material_dict.values()]

                beaker_plot = self._plot_axs[1, i].bar(material_names, material_amounts, color='blue')
                self._plot_axs[1, i].set_title(beakers[i].label)
                self._plot_axs[1, i].set_ylabel("Molar Amount")

                self._beaker_plots[i] = beaker_plot

            # draw on the existing graph
            self._plot_fig.canvas.draw()
            plt.pause(0.000001)
