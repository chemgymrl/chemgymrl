"""
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

Reaction Bench Engine Class

:title: reaction_bench_v0_engine

:author: Chris Beeler, Mitchell Shahen and Nicholas Paquin

:history: 21-05-2020

Class to initialize reaction vessels, prepare the reaction base environment, and designate actions to occur.
"""

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
import pickle
import os
import sys

sys.path.append("../../")  # to access chemistrylab
sys.path.append("../reactions/")  # to access all reactions
from chemistrylab.chem_algorithms.reward import ReactionReward
from chemistrylab.characterization_bench.characterization_bench import CharacterizationBench
from chemistrylab.chem_algorithms import vessel, util
from chemistrylab.reactions.get_reactions import convert_to_class
from chemistrylab.reactions.reaction_base import _Reaction

R = 0.008314462618  # Gas constant (kPa * m**3 * mol**-1 * K**-1)\


class ReactionBenchEnv(gym.Env):
    """
    Class to define elements of an engine to represent a reaction.
    """

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
        """
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
        """

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

        self.n_steps = input_parameters["n_steps"]  # Time steps per action
        self.dt = input_parameters["dt"]  # Time step of reaction (s)

        # initialize the reaction by specifying the reaction file to locate and acquire
        self.reaction = input_parameters["reaction"](
            reaction_file_identifier=input_parameters["reaction_file_identifier"],
            overlap=input_parameters["overlap"]
        )

        # Maximum time (s) (20 is the registered max_episode_steps)
        self.tmax = self.n_steps * self.dt * 20
        self.t = 0

        # initialize a step counter
        self.step_num = 1

        # prepare the initial vessel
        self.initial_materials = input_parameters["materials"]
        self.initial_solutes = input_parameters["solutes"]
        self.vessels = self._prepare_vessel(
            in_vessel_path=self.in_vessel_path,
            materials=input_parameters["materials"],
            solutes=input_parameters["solutes"]
        )

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
        self.char_bench = CharacterizationBench()
        absorb = self.char_bench.get_spectra(self.vessels, materials=self.reaction.materials)

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

        # set up plotting variables
        self._first_render = True
        self.plot_data_state = []
        self.plot_data_mol = []
        self.plot_data_concentration = []

        # set up the done parameter
        self.done = False

        # Finally, reset the environment
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
        """
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
        """

        # ----- ## CHECK PARAMETER TYPES ## ----- #

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
            "reaction": reaction,
            "reaction_file_identifier": reaction_file_identifier,
            "in_vessel_path": in_vessel_path,
            "out_vessel_path": out_vessel_path,
            "materials": materials,
            "solutes": solutes,
            "n_steps": n_steps,
            "dt": dt,
            "overlap": overlap
        }

        return input_parameters

    def update_vessel(self, new_vessel: vessel.Vessel):
        """
        Method to initialize a vessel and reset the environment to properly populate the vessel.

        Parameters
        ---------------
        `new_vessel` : `vessel.Vessel`
            The vessel that has been requested to be updated.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # set up a new, empty n array intended to contain the vessel materials
        new_n = np.zeros(self.reaction.nmax.shape[0], dtype=np.float32)

        # acquire the appropriate material dictionary from the inputted vessel
        mat_dict = util.convert_material_dict_units(new_vessel.get_material_dict())

        # iterate through the material dictionary populating the n array
        for i, mat in enumerate(self.reaction.materials):
            if mat in mat_dict:
                amount = mat_dict[mat][1]
                new_n[i] = amount

        # set the inputted vessel as the vessels variable
        self.vessels = new_vessel

        # modify the vessel appropriately and reset the reaction environment
        self.vessels = self.reaction.reset(self.vessels)

    @staticmethod
    def _prepare_materials(materials=[]):
        """
        Method to prepare a list of materials into a material dictionary.
        The provided materials are expected to be of the following form:
            materials = [
                {"Material": INSERT MATERIAL NAME, "Initial": INSERT INITIAL AMOUNT}
                {"Material": INSERT MATERIAL NAME, "Initial": INSERT INITIAL AMOUNT}
                .
                .
                .
            ]

        Parameters
        ---------------
        `materials` : `list` (default=`None`)
            A list of dictionaries including initial material names and amounts.

        Returns
        ---------------
        `material_dict` : `dict`
            A dictionary containing all the inputted materials, class representations, and their molar amounts.

        Raises
        ---------------
        None
        """

        # prepare lists to maintain the names and amounts of the materials
        material_names = []
        material_amounts = []

        # iterate through the provided list of materials and obtain the names and amounts
        for material_params in materials:
            material_names.append(material_params["Material"])
            material_amounts.append(material_params["Initial"])

        # acquire the material class representations from the material names
        material_classes = convert_to_class(materials=material_names)

        # create a dictionary to contain the material and its properties
        material_dict = {}

        # iterate through each of the material names, classes, and amounts lists;
        # it is assumed the materials are provided in units of mol
        for i, material in enumerate(material_names):
            material_dict[material] = [material_classes[i], material_amounts[i], 'mol']

        return material_dict

    @staticmethod
    def _prepare_solutes(material_dict=None, solutes=None):
        """
        Method to prepare a list of materials into a material dictionary.
        The provided materials are expected to be of the following form:
            materials = [
                {"Material": INSERT MATERIAL NAME, "Initial": INSERT INITIAL AMOUNT}
                {"Material": INSERT MATERIAL NAME, "Initial": INSERT INITIAL AMOUNT}
                .
                .
                .
            ]

        Parameters
        ---------------
        `solutes` : `list` (default=`None`)
            A list of dictionaries including initial solute names and amounts.

        Returns
        ---------------
        `solute_dict` : `dict`
            A dictionary containing all the inputted solutes, class representations, and their molar amounts.

        Raises
        ---------------
        None
        """

        solute_dict = {}
        for name, material in material_dict.items():
            if name not in solute_dict and material[0]()._solute:
                solute_dict[name] = {}
                for solvent in solutes:
                    solvent_class = convert_to_class([solvent['Material']])
                    solute_dict[name][solvent['Material']] = [solvent_class, solvent["Initial"], 'mol']

        return solute_dict

    def _prepare_vessel(self, in_vessel_path=None, materials=[], solutes=[]):
        """
        Method to prepare the initial vessel.

        Parameters
        ---------------
        `in_vessel_path` : `str` (default=`None`)
            A string indicating the path to a vessel intended to be loaded into the reaction bench environment.
        `materials` : `list` (default=`None`)
            A list of dictionaries including initial material names and amounts.
        `solutes` : `list` (default=`None`)
            A list of dictionaries including initial solute names and amounts.

        Returns
        ---------------
        `vessels` : `vessel.Vessel`
            A vessel object containing materials and solutes to be used in the reaction bench.

        Raises
        ---------------
        None
        """

        # initialize vessels by providing a empty default vessel or loading an existing saved vessel
        if in_vessel_path is None:
            # prepare the provided materials into a compatible material dictionary
            material_dict = self._prepare_materials(materials=materials)

            # prepare the solutes that have been provided
            solute_dict = self._prepare_solutes(material_dict=material_dict, solutes=solutes)

            # add the materials and solutes to an empty vessel
            vessels = vessel.Vessel(
                'default',
                materials=material_dict,
                solutes=solute_dict,
                default_dt=self.dt
            )
        else:
            with open(in_vessel_path, 'rb') as handle:
                vessels = pickle.load(handle)

        return vessels

    def _update_state(self):
        """
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
        """

        # get the vessel's thermodynamic properties
        T = self.vessels.get_temperature()
        V = self.vessels.get_volume()

        # acquire the absorption spectra for the materials in the vessel
        absorb = self.char_bench.get_spectra(self.vessels, materials=self.reaction.materials)

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
        """
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
        """

        self._first_render = True
        self.done = False

        # reset the modifiable state variables
        self.t = 0.0
        self.step_num = 1

        # reinitialize the reaction class
        self.vessels = self._prepare_vessel(in_vessel_path=None, materials=self.initial_materials,
                                            solutes=self.initial_solutes)

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
            [0.0],  # time
            [(Ti - Tmin) / (Tmax - Tmin)],  # normalized temperature
            [(Vi - Vmin) / (Vmax - Vmin)],  # normalized volume
            [total_pressure / Pmax],  # normalized pressure
            [0.0],  # time
            [(Ti - Tmin) / (Tmax - Tmin)],  # normalized temperature
            [(Vi - Vmin) / (Vmax - Vmin)],  # normalized volume
            [total_pressure / Pmax]  # normalized pressure
        ]
        self.plot_data_mol = []
        self.plot_data_concentration = []

        # [0] is time, [1] is Temperature, [2:] is each species
        C = self.vessels.get_concentration(materials=self.reaction.materials)
        for i in range(self.reaction.nmax.shape[0]):
            self.plot_data_mol.append([self.reaction.n[i]])
            self.plot_data_concentration.append([C[i]])

        # update the state variables with the initial values
        self._update_state()

        return self.state

    def step(self, action):
        """
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
        """

        # the reward for this step is set to 0 initially
        reward = 0.0

        # pass the action and vessel to the reaction base class's perform action function
        self.vessels = self.reaction.perform_action(action, self.vessels, self.t, self.n_steps, self.step_num)

        # Increase time by time step
        self.t += self.n_steps * self.dt

        # call some function to get plotting data
        self.plot_data_state, self.plot_data_mol, self.plot_data_concentration = self.reaction.plotting_step(
            self.t, self.tmax, self.vessels
        )

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
        """
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
        """

        if model == 'human':
            self.human_render()
        elif model == 'full':
            self.full_render()
        else:
            print("Incorrect Render Model: {}".format(model))

    def human_render(self, model='plot'):
        """
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
        """
        wave_max = self.char_bench.params['spectra']['range_ir'][1]
        wave_min = self.char_bench.params['spectra']['range_ir'][0]
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

        # dict of wave data passed to plotting method in reaction
        wave_data_dict = {"spectra_len": spectra_len,
                          "absorb": absorb,
                          "wave": wave,
                          "wave_min": wave_min,
                          "wave_max": wave_max
                          }

        # call render function in reaction base
        self.reaction.plot_human_render(self._first_render, wave_data_dict, self.plot_data_state)

        if self._first_render:
            self._first_render = False

    def full_render(self, model='plot'):
        """
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
        """
        wave_max = self.char_bench.params['spectra']['range_ir'][1]
        wave_min = self.char_bench.params['spectra']['range_ir'][0]

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

        # dict of wave data passed to plotting method in reaction
        wave_data_dict = {"spectra_len": spectra_len,
                          "absorb": absorb,
                          "wave": wave,
                          "wave_min": wave_min,
                          "wave_max": wave_max
                          }

        # call render function in reaction base
        self.reaction.plot_full_render(
            self._first_render,
            wave_data_dict,
            self.plot_data_state,
            self.plot_data_mol,
            self.plot_data_concentration,
            self.n_steps,
            self.vessels,
            self.step_num
        )

        # get the spectral data peak and dashed spectral lines
        peak = self.char_bench.get_spectra_peak(
            self.vessels,
            materials=self.reaction.materials
        )
        dash_spectra = self.char_bench.get_dash_line_spectra(
            self.vessels,
            materials=self.reaction.materials
        )

        # The first render is required to initialize the figure
        if self._first_render:
            self._first_render = False
