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
            target_material=None
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

        return self.done
