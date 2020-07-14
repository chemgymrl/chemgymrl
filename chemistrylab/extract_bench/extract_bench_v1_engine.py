'''
Module defining the ExtractWorld Engine

:title: extractworld_v1_engine.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-06-24
'''

# pylint: disable=no-member
# pylint: disable=unsubscriptable-object

# import external modules
import cmocean
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys

# import local modules
sys.path.append("../../") # access chemistrylab
from chemistrylab.chem_algorithms import util
from chemistrylab.extract_algorithms.extractions import water_oil_v1, wurtz_v0
from chemistrylab.extract_algorithms import separate

# a dictionary of available extractions
extraction_dict = {
    'water_oil': water_oil_v1,
    "wurtz": wurtz_v0
}

class ExtractBenchEnv(gym.Env):
    '''
    Class object defining the ExtractWorld Engine

    Parameters
    ---------------
    `n_steps` : `int` (default=`100`)
        The number of steps in an episode.
    `dt` : `float` (default=`0.01`)
        The default time-step.
    `extraction` : `str` (default="")
        The name/title of the extraction.
    `n_vessel_pixels` : `int` (default=`100`)
        The number of pixels in a vessel.
    `max_valve_speed` : `int` (default=`100`)
        The maximal amount of material flowing through the valve in a given time-step.
    `extraction_vessel` : `vessel` (default=`None`)
        A vessel object containing state variables, materials, solutes, and spectral data.
    `solute` : `str` (default=`None`)
        The name of the added solute.
    `target_material` : `str` (default=`None`)
        The name of the required output material designated as reward.

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
        dt=0.01,
        extraction="",
        n_vessel_pixels=100,
        max_valve_speed=10,
        extraction_vessel=None,
        solute=None,
        target_material=None
    ):
        '''
        Constructor class method for the Extract Bench Environment
        '''

        self.n_steps = n_steps
        self.dt = dt
        self.n_vessel_pixels = n_vessel_pixels
        self.max_valve_speed = max_valve_speed
        self.extraction_vessel = extraction_vessel
        # print(self.extraction_vessel._solute_dict)
        self.solute = solute
        self.target_material = target_material

        self.extraction = extraction_dict[extraction].Extraction(
            extraction_vessel=self.extraction_vessel,
            n_vessel_pixels=self.n_vessel_pixels,
            max_valve_speed=self.max_valve_speed,
            solute=self.solute,
            target_material=self.target_material
        )

        self.vessels = None
        self.external_vessels = None
        self.state = None

        # self.observation_space = self.extraction.get_observation_space()
        self.action_space = self.extraction.get_action_space()
        self.done = False
        self._first_render = True
        self._plot_fig = None
        self._plot_axs = None

        # Reset the environment to start
        # self.reset()

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

        self.done = False
        self._first_render = True
        self.vessels, self.external_vessels, self.state = self.extraction.reset(
            extraction_vessel=self.extraction_vessel
        )

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

        # define the necessary parameters
        vessels = self.vessels
        ext_vessels = self.external_vessels
        done = self.done

        # perform the inputted action in the extraction module
        vessels, ext_vessels, reward, done = self.extraction.perform_action(
            vessels=vessels,
            external_vessels=ext_vessels,
            action=action
        )

        # re-define self variables for future use
        self.vessels = vessels
        self.external_vessels = ext_vessels
        self.done = done

        # determine the state after performing the action
        self.state = util.generate_state(
            self.vessels,
            max_n_vessel=self.extraction.n_total_vessels
        )

        # document the most recent step and determine if future steps are necessary
        self.n_steps -= 1
        if self.n_steps == 0:
            self.done = True
            reward = self.extraction.done_reward(self.vessels[2])

        return self.state, reward, self.done, {}

    def render(self, model='human'):
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

        position_separate = []
        for vessel_obj in self.vessels:
            position, __ = vessel_obj.get_position_and_variance()
            position_separate.append(
                np.zeros((len(position), separate.x.shape[0]), dtype=np.float32)
            )

        for i, arr in enumerate(position_separate):
            position, var = self.vessels[i].get_position_and_variance(dict_or_list='dict')
            t = -1.0 * np.log(var * np.sqrt(2.0 * np.pi))
            j = 0

            for layer in position:
                # Loop over each phase
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
                [x[2] for x in self.state]
            ),
            (
                self.extraction.n_total_vessels,
                self.extraction.n_vessel_pixels,
                1
            )
        )

        if self._first_render:
            plt.close('all')
            plt.ion()
            self._plot_fig, self._plot_axs = plt.subplots(
                self.extraction.n_total_vessels,
                2,
                figsize=(12, 6),
                gridspec_kw={'width_ratios': [3, 1]}
            )

            for i, arr in enumerate(position_separate):
                position, __ = self.vessels[i].get_position_and_variance()
                j = 0

                for layer in position:
                    if layer == 'Air':
                        color = self.vessels[i].Air.get_color()
                    else:
                        color = self.vessels[i].get_material_color(layer)

                    color = cmocean.cm.delta(color)
                    self._plot_axs[i, 0].plot(separate.x, arr[j], label=layer, c=color)
                    j += 1

                self._plot_axs[i, 0].set_xlim([separate.x[0], separate.x[-1]])
                self._plot_axs[i, 0].set_ylim([0, self.vessels[i].v_max])
                self._plot_axs[i, 0].set_xlabel('Separation')
                self._plot_axs[i, 0].legend()

                __ = self._plot_axs[i, 1].pcolormesh(
                    Ls[i],
                    vmin=0,
                    vmax=1,
                    cmap=cmocean.cm.delta
                )

                self._plot_axs[i, 1].set_xticks([])
                self._plot_axs[i, 1].set_ylabel('Height')
                # self._plot_axs[i, 1].colorbar(mappable)
                self._plot_fig.canvas.draw()
                plt.show()

                self._first_render = False
        else:
            for i, arr in enumerate(position_separate):
                self._plot_axs[i, 0].clear()
                position, __ = self.vessels[i].get_position_and_variance()

                j = 0
                for layer in position:
                    if layer == 'Air':
                        color = self.vessels[i].Air.get_color()
                    else:
                        color = self.vessels[i].get_material_color(layer)

                    color = cmocean.cm.delta(color)
                    self._plot_axs[i, 0].plot(separate.x, arr[j], label=layer, c=color)
                    j += 1

                self._plot_axs[i, 0].set_xlim([separate.x[0], separate.x[-1]])
                self._plot_axs[i, 0].set_ylim([0, self.vessels[i].v_max])
                self._plot_axs[i, 0].set_xlabel('Separation')
                self._plot_axs[i, 0].legend()
                __ = self._plot_axs[i, 1].pcolormesh(
                    Ls[i],
                    vmin=0,
                    vmax=1,
                    cmap=cmocean.cm.delta
                )
                # self._plot_axs[i, 1].colorbar(mappable)
                self._plot_fig.canvas.draw()
                # plt.show() # adding this causes the plot to be unresponsive when updating
                self._first_render = False
