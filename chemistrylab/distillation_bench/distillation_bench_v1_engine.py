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
from chemistrylab.chem_algorithms import util
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

        # set necessary variables
        self.n_steps = n_steps
        self.boil_vessel = boil_vessel
        self.target_material = target_material

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

        # set up the intended path for the final distillation vessel to be stored
        self.vessel_path = out_vessel_path

    def _calc_reward(self):
        '''
        Method to calculate the generated reward in every vessel in `self.vessels`.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `total_reward` : `float`
            The sum of reward found in each vessel.

        Raises
        ---------------
        None
        '''

        total_reward = 0

        # search every vessel's material_dict for the target material
        for vessel_obj in self.vessels:
            material_names = vessel_obj._material_dict.keys()

            if self.target_material in material_names:
                total_reward += self.distillation.done_reward(beaker=vessel_obj)

        if total_reward == 0:
            print("Error: No target material found in any vessel!")

        return total_reward

    def _save_vessel(self, vessel):
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
            filename = "vessel_distillation.pickle"
            open_file = os.path.join(file_directory, filename)

        # delete any existing vessel files to ensure the vessel is saved as intended
        num = 0
        while os.path.exists(open_file):
            # rename the filename by adding a number
            start_filename = filename.split(".")[0]
            new_start_filename = "{}_{}".format(start_filename, num)
            new_filename = "{}.pickle".format(new_start_filename)

            # recreate the open_file variable
            open_file = os.path.join(file_directory, new_filename)

            # increase the arbitrary marker by 1
            num += 1

        # open the intended vessel file and save the vessel as a pickle file
        with open(open_file, 'wb') as vessel_file:
            pickle.dump(vessel, vessel_file)

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

        # reset done and plotting parameters to initial values
        self.done = False
        self._first_render = True

        # acquire the initial vessels and state from the distillation class
        vessels, state = self.distillation.reset(
            boil_vessel=self.boil_vessel
        )

        # redefine the existing vessels and state variables
        self.vessels = vessels
        self.state = state

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

        # generate the state variable
        self.state = util.generate_state(
            vessel_list=self.vessels,
            max_n_vessel=self.distillation.n_total_vessels
        )

        # update the number of steps left to complete
        self.n_steps -= 1
        if self.n_steps == 0:
            self.done = True

            # after the last step, calculate the final reward
            reward = self._calc_reward()

            # option to save any final vessels here
            # self._save_vessel(vessel=self.vessels[0])

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

