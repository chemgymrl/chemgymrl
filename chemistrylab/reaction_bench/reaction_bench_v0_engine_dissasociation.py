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

sys.path.append("../../") # to access chemistrylab
sys.path.append("../reactions/") # to access all reactions
from chemistrylab.chem_algorithms.reward import ReactionReward
from chemistrylab.chem_algorithms import vessel
from chemistrylab.reactions.decomp_reaction import Reaction

R = 0.008314462618 # Gas constant (kPa * m**3 * mol**-1 * K**-1)
wave_max = 800
wave_min = 200

class ReactionBenchEnvDissasociation(gym.Env):
    '''
    Class to define elements of an engine to represent a reaction.
    '''

    def __init__(
            self,
            reaction=Reaction,
            in_vessel_path=None,
            out_vessel_path=None,
            materials=None,
            solutes=None,
            desired="",
            n_steps=50,
            dt=0.01,
            Ti=300.0,
            Tmin=250.0,
            Tmax=500.0,
            dT=50.0,
            Vi=0.002,
            Vmin=0.001,
            Vmax=0.005,
            dV=0.0005,
            Pmax=1,
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
        `desired` : `str` (default="")
            A string indicating the desired output material.
        `n_steps` : `int` (default=`50`)
            The number of time steps to be taken during each action.
        `dt` : `float` (default=`0.01`)
            The amount of time taken in each time step.
        `Ti` : `float` (default=`300.0`)
            The initial temperature of the system in Kelvin.
        `Tmin` : `float` (default=`250.0`)
            The minimal temperature of the system in Kelvin.
        `Tmax` : `float` (default=`500.0`)
            The maximal temperature of the system in Kelvin.
        `dT` : `float` (default=`50.0`)
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
            in_vessel_path=in_vessel_path,
            out_vessel_path=out_vessel_path,
            materials=materials,
            solutes=solutes,
            desired=desired,
            n_steps=n_steps,
            dt=dt,
            Ti=Ti,
            Tmin=Tmin,
            Tmax=Tmax,
            dT=dT,
            Vi=Vi,
            Vmin=Vmin,
            Vmax=Vmax,
            dV=dV,
            overlap=overlap
        )

        # set the input vessel path
        self.in_vessel_path = input_parameters["in_vessel_path"]

        # set the output vessel path
        self.out_vessel_path = input_parameters["out_vessel_path"]

        # set initial material, solute, and overlap parameters
        materials = input_parameters["materials"]
        solutes = input_parameters["solutes"]
        overlap = input_parameters["overlap"]

        # set the remaining input parameters
        self.desired = input_parameters["desired"] # the desired material
        self.n_steps = input_parameters["n_steps"] # Time steps per action
        self.dt = input_parameters["dt"] # Time step of reaction (s)
        self.Ti = input_parameters["Ti"] # Initial temperature (K)
        self.Tmin = input_parameters["Tmin"] # Minimum temperature of the system (K)
        self.Tmax = input_parameters["Tmax"] # Maximum temperature of the system (K)
        self.dT = input_parameters["dT"] # Maximum change in temperature per action (K)
        self.Vi = input_parameters["Vi"] # Initial Volume (L)
        self.Vmin = input_parameters["Vmin"] # Minimum Volume of the system (L)
        self.Vmax = input_parameters["Vmax"] # Maximum Volume of the system (L)
        self.dV = input_parameters["dV"] # Maximum change in Volume per action (L)
        self.Pmax = Pmax

        # initialize the reaction
        self.reaction = input_parameters["reaction"](
            overlap=overlap,
            materials=materials,
            solutes=solutes,
            desired=self.desired
        )

        # Maximum time (s) (20 is the registered max_episode_steps)
        self.tmax = self.n_steps * dt * 20

        # initialize a step counter
        self.step_num = 1

        # initialize vessels by providing a empty default vessel or loading an existing saved vessel
        self.n_init = np.zeros(self.reaction.nmax.shape[0], dtype=np.float32)
        if self.in_vessel_path is None:
            self.vessels = vessel.Vessel(
                'default',
                temperature=self.Ti,
                v_max=self.Vmax,
                v_min=self.Vmin,
                Tmax=self.Tmax,
                Tmin=self.Tmin,
                default_dt=self.dt
            )

            '''
            for i in range(self.reaction.initial_in_hand.shape[0]):
                self.n_init[i] = self.reaction.initial_in_hand[i]
            '''
        else:
            with open(self.in_vessel_path, 'rb') as handle:
                v = pickle.load(handle)
                self.vessels = v

            '''
            for i in range(self.n_init.shape[0]):
                material_name = self.reaction.labels[i]
                self.n_init[i] = self.vessels._material_dict[material_name][1]
            '''

        # set up a state variable
        self.state = None

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

    @staticmethod
    def _validate_parameters(
            reaction=None,
            in_vessel_path="",
            out_vessel_path="",
            materials=None,
            solutes=None,
            desired="",
            n_steps=0,
            dt=0.0,
            Ti=0.0,
            Tmin=0.0,
            Tmax=0.0,
            dT=0.0,
            Vi=0.0,
            Vmin=0.0,
            Vmax=0.0,
            dV=0.0,
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
        `desired` : `str` (default="")
            A string indicating the desired output material.
        `n_steps` : `int` (default=`0`)
            The number of time steps to be taken during each action.
        `dt` : `float` (default=`0.0`)
            The amount of time taken in each time step.
        `Ti` : `float` (default=`1.0`)
            The initial temperature of the system in Kelvin.
        `Tmin` : `float` (default=`1.0`)
            The minimal temperature of the system in Kelvin.
        `Tmax` : `float` (default=`1.0`)
            The maximal temperature of the system in Kelvin.
        `dT` : `float` (default=`0.0`)
            The maximal allowed temperature change, in Kelvin, for a single action.
        `Vi` : `float` (default=`1.0`)
            The initial volume of the system in litres.
        `Vmin` : `float` (default=`1.0`)
            The minimal volume of the system in litres.
        `Vmax` : `float` (default=`1.0`)
            The maximal volume of the system in litres.
        `dV` : `float` (default=`0.0`)
            The maximal allowed volume change, in litres, for a single action.
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

        # ensure the desired material parameter is a string
        if not isinstance(desired, str):
            print("Invalid `desired` type. The default will be provided.")
            desired = ""

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

        # the initial temperature parameter must be a non-zero, non-negative floating point value
        if any([
                not isinstance(Ti, float),
                Ti <= 0.0
        ]):
            print("Invalid 'Initial Temperature' type. The default will be provided.")
            Ti = 1.0

        # the minimum temperature parameter must be a non-zero, non-negative floating point value
        if any([
                not isinstance(Tmin, float),
                Tmin <= 0.0
        ]):
            print("Invalid 'Minimum Temperature' type. The default will be provided.")
            Tmin = 1.0

        # the maximum temperature parameter must be a non-zero, non-negative floating point value
        if any([
                not isinstance(Tmax, float),
                Tmax <= 0.0
        ]):
            print("Invalid 'Maximum Temperature' type. The default will be provided.")
            Tmax = 1.0

        # the temperature increment parameter must be a non-zero, non-negative floating point value
        if any([
                not isinstance(dT, float),
                dT <= 0.0
        ]):
            print("Invalid 'Temperature Increment' type. The default will be provided.")
            dT = 1.0

        # the initial volume parameter must be a non-zero, non-negative floating point value
        if any([
                not isinstance(Vi, float),
                Vi <= 0.0
        ]):
            print("Invalid 'Initial Volume' type. The default will be provided.")
            Vi = 1.0

        # the minimum volume parameter must be a non-zero, non-negative floating point value
        if any([
                not isinstance(Vmin, float),
                Vmin <= 0.0
        ]):
            print("Invalid 'Minimum Volume' type. The default will be provided.")
            Vmin = 1.0

        # the maximum volume parameter must be a non-zero, non-negative floating point value
        if any([
                not isinstance(Vmax, float),
                Vmax <= 0.0
        ]):
            print("Invalid 'Maximum Volume' type. The default will be provided.")
            Vmax = 1.0

        # the volume increment parameter must be a non-zero, non-negative floating point value
        if any([
                not isinstance(dV, float),
                dV <= 0.0
        ]):
            print("Invalid 'Volume Increment' type. The default will be provided.")
            dV = 1.0

        # the overlap parameter must be a boolean
        if not isinstance(overlap, bool):
            print("Invalid 'Overlap Spectra' type. The default will be provided.")
            overlap = False

        ## ----- ## CHECK THERMODYNAMIC VARIABLES ## ----- ##

        # the minimum temperature parameter must be smaller than or equal to the maximum temperature
        if Tmin > Tmax:
            Tmin = Tmax

        # the maximum temperature parameter must be larger than or equal to the minimum volume
        if Tmax < Tmin:
            Tmax = Tmin

        # the initial temperature must be between the minimal and maximal temperature parameters
        if not Tmin <= Ti <= Tmax:
            Ti = Tmin + ((Tmax - Tmin) / 2)

        # the minimum volume parameter must be smaller than or equal to the maximum volume
        if Vmin > Vmax:
            Vmin = Vmax

        # the maximum volume parameter must be larger than or equal to the minimum volume
        if Vmax < Vmin:
            Vmax = Vmin

        # the initial temperature must be between the minimal and maximal temperature parameters
        if not Vmin <= Vi <= Vmax:
            Vi = Vmin + ((Vmax - Vmin) / 2)

        # collect all the input parameters in a labelled dictionary
        input_parameters = {
            "reaction" : reaction,
            "in_vessel_path" : in_vessel_path,
            "out_vessel_path" : out_vessel_path,
            "materials" : materials,
            "solutes" : solutes,
            "desired" : desired,
            "n_steps" : n_steps,
            "dt" : dt,
            "Ti" : Ti,
            "Tmin" : Tmin,
            "Tmax" : Tmax,
            "dT" : dT,
            "Vi" : Vi,
            "Vmin" : Vmin,
            "Vmax" : Vmax,
            "dV" : dV,
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
        temperature = self.T
        volume = self.V
        pressure = self.P

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
            default_dt=self.dt
        )
        new_vessel.temperature = temperature
        new_vessel._material_dict = new_material_dict
        new_vessel._solute_dict = new_solute_dict
        new_vessel.volume = volume
        new_vessel.pressure = pressure

        # replace the old vessel with the updated version
        self.vessels = new_vessel

    def _save_vessel(self, vessel_rootname=""):
        '''
        Method to save a vessel as a pickle file.

        Parameters
        ---------------
        `vessel_rootname` : `str` (default="")
            The intended file root of the vessel (no extension; not the absolute path).

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # use the vessel path that was given upon initialization or during parameter verification
        output_directory = self.out_vessel_path
        vessel_filename = "{}.pickle".format(vessel_rootname)
        open_file = os.path.join(output_directory, vessel_filename)

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
        self.P = 1.0 * self.vessels.get_pressure()

        # reinitialize the reaction class
        self.reaction.reset(n_init=self.n_init)

        # save the necessary temperatures, volumes, and pressure
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

        # the reward for this step is set to 0 initially
        reward = 0.0

        # the first two action variables are changes to temperature and volume, respectively;
        # these changes need to be scaled according to the maximum allowed change, dT and dV;
        # action[0] = 1.0 gives a temperature change of dT, while action[0] = 0.0 corresponds to -dT
        # action[1] = 1.0 gives a temperature change of dV, while action[1] = 0.0 corresponds to -dV
        temperature_change = 2.0 * (action[0] - 0.5) * self.dT # in Kelvin
        volume_change = 2.0 * (action[1] - 0.5) * self.dV # in Litres

        # the remaining action variables are the proportions of reactants to be added
        add_reactant_proportions = action[2:]

        # set up a variable to contain the amount of change in each reactant
        delta_n_array = np.zeros(len(self.reaction.cur_in_hand))

        # scale the action value by the amount of reactant available
        for i, reactant_amount in enumerate(self.reaction.cur_in_hand):
            # determine how much of the available reactant is to be added
            proportion = add_reactant_proportions[i]

            # determine the amount of the reactant available
            amount_available = reactant_amount

            # determine the amount of reactant to add (in mol)
            delta_n = proportion * amount_available

            # add the overall change in materials to the array of molar amounts
            delta_n_array[i] = delta_n

        for i in range(self.n_steps):
            # split the overall temperature change into increments
            self.T += temperature_change / self.n_steps
            self.T = np.min([
                np.max([self.T, self.vessels.get_Tmin()]),
                self.vessels.get_Tmax()
            ])

            # split the overall volume change into increments
            self.V += volume_change / self.n_steps
            self.V = np.min([
                np.max([self.V, self.vessels.get_min_volume()]),
                self.vessels.get_max_volume()
            ])

            # split the overall molar changes into increments
            for i, delta_n in enumerate(delta_n_array):
                dn = delta_n / self.n_steps
                self.reaction.n[i] += dn
                self.reaction.cur_in_hand[i] -= dn

                # set a penalty for trying to add unavailable material (tentatively set to 0)
                # reward -= 0

                # if the amount that is in hand is below a certain threshold set it to 0
                if self.reaction.cur_in_hand[i] < 1e-8:
                    self.reaction.cur_in_hand[i] = 0.0

            # perform the reaction and update the molar concentrations of the reactants and products
            self.reaction.update(
                self.T,
                self.V,
                self.vessels.get_defaultdt()
            )

            # Increase time by time step
            self.t += self.dt

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
            self.P = self.reaction.get_total_pressure(self.V, self.T)
            self.plot_data_state[3].append(
                self.P/self.vessels.get_pmax()
            )

            # calculate and record the molar concentrations of the reactants and products
            C = self.reaction.get_concentration(self.V)
            for j in range(self.reaction.n.shape[0]):
                self.plot_data_mol[j].append(self.reaction.n[j])
                self.plot_data_concentration[j].append(C[j])

        # update the state variables with the changes during the last step
        self._update_state()

        # update the vessels variable
        self._update_vessel()

        # calculate the reward for this step
        reward = ReactionReward(
            vessel=self.vessels,
            desired_material=self.desired
        ).calc_reward()

        # ---------- ADD ---------- #
        # add option to save or print intermediary vessel

        # save the vessel when the final step is complete
        if any([self.done, self.step_num == 20]):
            self._save_vessel(vessel_rootname="react_vessel")

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

        reactant_labels = []
        for reactant in self.reaction.get_ni_label():
            name = reactant.replace("[", "").replace("]", "")
            short_name = name[0:5]
            reactant_labels.append(short_name)

        label_list = ['t', 'T', 'V', 'P'] + reactant_labels

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

        reactant_labels = []
        for reactant in self.reaction.get_ni_label():
            name = reactant.replace("[", "").replace("]", "")
            short_name = name[0:4]
            reactant_labels.append(short_name)

        label_list = ['t', 'T', 'V', 'P'] + reactant_labels

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
