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

Module to describe the extraction of a desired material generated in a Wurtz reaction.
:title: wurtz_v0.py
:author: Mitchell Shahen
:history: 2020-06-27
Available Actions for this Extraction Experiment are included below.
0: Valve (Speed multiplier, relative to max_valve_speed)
1: Mix ExV (mixing coefficient, *-1 when passed into mix function)
2: Pour B1 into ExV (Volume multiplier, relative to max_vessel_volume)
3: Pour B2 into ExV (Volume multiplier, relative to max_vessel_volume)
4: Pour ExV into B2 (Volume multiplier, relative to max_vessel_volume)
5: Pour S1 into ExV (Volume multiplier, relative to max_vessel_volume)
6: Pour S2 into ExV (Volume multiplier, relative to max_vessel_volume)
7: Done (Value doesn't matter)
"""

# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-position

# import external modules
from copy import deepcopy
import sys
import numpy as np

# import OpenAI Gym
import gym

# import local modules
sys.path.append("../../../") # to access chemistrylab
from chemistrylab.chem_algorithms import util, vessel
from chemistrylab.reactions.get_reactions import convert_to_class

class Extraction:
    '''
    Class object for a Wurtz extraction experiment.
    Parameters
    ---------------
    `extraction_vessel` : `vessel` (default=`None`)
        A vessel object containing state variables, materials, solutes, and spectral data.
    `solvents` : `str` (default=`None`)
        The name of the added solvents.
    `target_material` : `str` (default=`None`)
        The name of the required output material designated as reward.
    `nempty_vessels` : `int` (default=`2`)
        The number of empty vessels to add.
    `solvent_volume` : `int` (default=`1000000000`)
        The amount of `solvent` available to add during the extraction.
    `dt` : `float` (default=`0.01`)
        The default time-step.
    `max_vessel_volume` : `float` (default=`1000.0`)
        The maximal volume of an added vessel.
    `n_vessel_pixels` : `int` (default=`100`)
        The number of pixels in a vessel.
    `max_valve_speed` : `int` (default=`10`)
        The maximal amount of material flowing through the valve in a given time-step.
    `n_actions` : `int` (default=`8`)
        The number of actions included in this extraction experiment.
    Returns
    ---------------
    None
    Raises
    ---------------
    None
    '''

    def __init__(
            self,
            extraction_vessel,
            solvents,
            target_material,
            n_empty_vessels=2,  # number of empty vessels
            solvent_volume=1000000000,  # the amount of solvent available (unlimited)
            dt=0.01,  # time for each step (time for separation)
            max_vessel_volume=1.0,  # max volume of empty vessels in L
            n_vessel_pixels=100,  # number of pixels for each vessel
            max_valve_speed=10,  # maximum draining speed (pixels/step)
            n_actions=8,
            extractor=None
    ):
        '''
        Constructor class for the Wurtz Extraction class
        '''

        # set self variable
        self.dt = dt
        self.n_actions = n_actions
        self.max_vessel_volume = max_vessel_volume
        self.solvent_volume = solvent_volume
        self.n_empty_vessels = n_empty_vessels
        self.n_total_vessels = n_empty_vessels + 1  # (empty + input)
        self.n_vessel_pixels = n_vessel_pixels
        self.max_valve_speed = max_valve_speed

        self.solvents = solvents
        self.target_material = target_material
        self.target_material_init_amount = extraction_vessel.get_material_amount(target_material)

    def get_observation_space(self, targets):
        obs_low = np.zeros((self.n_total_vessels, self.n_vessel_pixels + len(targets)), dtype=np.float32)
        obs_high = 1.0 * np.ones((self.n_total_vessels, self.n_vessel_pixels + len(targets)), dtype=np.float32)
        observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        return observation_space

    def get_action_space(self):
        """
        Method to describe the actions index and multipliers available.
        Parameters
        ---------------
        None
        Returns
        ---------------
        `action_space` : `MultiDiscrete`
            A MultiDiscrete object indicating the options for selecting actions and multipliers.
        Raises
        ---------------
        None
        """

        action_space = gym.spaces.MultiDiscrete([self.n_actions, 5])

        return action_space

    def reset(self, extraction_vessel, targets):
        """
        Method to reset the environment.
        Parameters
        ---------------
        `extraction_vessel` : `vessel` (default=`None`)
            A vessel object containing state variables, materials, solutes, and spectral data.
        Returns
        ---------------
        `vessels` : `list`
            A list of all the vessel objects that contain materials and solutes.
        `external_vessels` : `list`
            A list of the external vessels, beakers, to be used in the extraction.
        `state` : `np.array`
            An array containing state variables, material concentrations, and spectral data.
        Raises
        ---------------
        None
        """

        # copy initial extraction vessel into a list of vessels
        vessels = [deepcopy(extraction_vessel)]
        
        # create all the necessary beakers and add them to the list
        for i in range(self.n_empty_vessels):
            temp_vessel = vessel.Vessel(
                label='beaker_{}'.format(i + 1),
                v_max=self.max_vessel_volume,
                default_dt=0.05,
                n_pixels=self.n_vessel_pixels
            )
            vessels.append(temp_vessel)

        # generate a list of external vessels to contain solvents
        external_vessels = []

        for mat in self.solvents:
            # generate a vessel to contain the main solvent
            solvent_vessel = vessel.Vessel(
                label='solvent_vessel0',
                v_max=self.solvent_volume,
                n_pixels=self.n_vessel_pixels,
                settling_switch=False,
                layer_switch=False,
            )

            # create the material dictionary for the solvent vessel
            solvent_material_dict = {}
            solvent_class = convert_to_class(materials=[mat])[0]()
            solvent_material_dict[mat] = [solvent_class, self.solvent_volume]

            # check for overflow
            solvent_material_dict, _, _ = util.check_overflow(
                material_dict=solvent_material_dict,
                solute_dict={},
                v_max=solvent_vessel.get_max_volume()
            )

            # instruct the vessel to update its material dictionary
            event = ['update material dict', solvent_material_dict]
            solvent_vessel.push_event_to_queue(feedback=[event], dt=0)

            # add the main solvent vessel to the list of external vessels
            external_vessels.append(solvent_vessel)

        state = np.zeros((self.n_total_vessels, self.n_vessel_pixels + len(targets)), dtype=np.float32)

        # generate the state
        state[:, :self.n_vessel_pixels] = util.generate_layers_obs(
            vessel_list=vessels,
            max_n_vessel=self.n_total_vessels,
            n_vessel_pixels=self.n_vessel_pixels
        )

        targ_ind = targets.index(self.target_material)
        state[:, self.n_vessel_pixels + targ_ind] += 1

        return vessels, external_vessels, state

    def perform_action(self, vessels, ext_vessel, action):
        """
        Method to perform the action designated by `action` and update
        the state, vessels, external vessels, and generate reward.
        Parameters
        ---------------
        `vessels` : `list`
            A list of all the vessel objects that contain materials and solutes.
        `external_vessels` : `list`
            A list of the external vessels, beakers, to be used in the extraction.
        `action` : `list`
            A list of two numbers indicating the index of an action to
            perform and a multiplier used when performin the action.
        Returns
        ---------------
        `vessels` : `list`
            A list of all the vessel objects that contain materials and solutes.
        `external_vessels` : `list`
            A list of the external vessels, beakers, to be used in the extraction.
        `reward` : `float`
            The amount of the target material that has been generated by the most recent action.
        `done` : `bool`
            A boolean indicating if all of the required actions have now been completed.
        Raises
        ---------------
        None
        """

        # default reward for each step
        reward = 0

        # set the completed variable
        done = False

        # deconstruct the action
        do_action = int(action[0])
        multiplier = (action[1]) / 4 if action[1] != 0 else 0

        # if the multiplier is 0, push an empty list of events to the vessel queue
        if all([multiplier == 0, do_action != 7]):
            for vessel_obj in vessels:
                __ = vessel_obj.push_event_to_queue(dt=self.dt)
        else:
            # obtain the necessary vessels
            extract_vessel = vessels[0]
            beaker_1 = vessels[1]
            beaker_2 = vessels[2]
            solvent_vessel1 = ext_vessel[0]
            solvent_vessel2 = ext_vessel[1]
            
            print(do_action,multiplier)
            # Open Valve (Speed multiplier)
            if do_action == 0:
                # calculate the number of pixels being drained
                drained_pixels = multiplier * self.max_valve_speed

                # drain the extraction vessel into the first beaker;
                event = ['drain by pixel', beaker_1, drained_pixels]

                # push the event to the extraction vessel
                reward = extract_vessel.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the second beaker
                __ = beaker_2.push_event_to_queue(dt=self.dt)

            # Mix the Extraction Vessel
            if do_action == 1:
                # mix the extraction vessel
                event = ['mix', -multiplier]

                # push the event to the extraction vessel
                reward = extract_vessel.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to either beaker
                __ = beaker_1.push_event_to_queue(dt=self.dt)
                __ = beaker_2.push_event_to_queue(dt=self.dt)

            # pour Beaker 1 into the Extraction Vessel
            if do_action == 2:
                # determine the volume to pour from the first beaker into the extraction vessel
                d_volume = beaker_1.get_max_volume() * multiplier

                # push the event to the first beaker
                event = ['pour by volume', extract_vessel, d_volume]
                reward = beaker_1.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the second beaker
                __ = beaker_2.push_event_to_queue(dt=self.dt)

            # Pour Beaker 2 into the Extraction Vessel
            if do_action == 3:
                # determine the volume to pour from the second beaker into the extraction vessel
                d_volume = beaker_2.get_max_volume() * multiplier

                # push the event to the second beaker
                event = ['pour by volume', extract_vessel, d_volume]
                reward = beaker_2.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the first beaker
                beaker_1.push_event_to_queue(dt=self.dt)

            # Pour the Extraction Vessel into Beaker 2
            if do_action == 4:
                # determine the volume to pour from the extraction vessel into the second beaker
                d_volume = 1.0 * multiplier # use the default volume for the extraction vessel

                # push the event to the extraction vessel
                event = ['pour by volume', beaker_2, d_volume]
                reward = extract_vessel.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the first beaker
                beaker_1.push_event_to_queue(dt=self.dt)

            # pour the (first) Solute Vessel into the Extraction Vessel
            if do_action == 5:
                # determine the volume to pour from the solvent vessel into the extraction vessel
                d_volume = extract_vessel.get_max_volume() * multiplier

                # push the event to the solvent vessel
                event = ['pour by volume', extract_vessel, d_volume]
                reward = solvent_vessel1.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to either of the beakers
                beaker_1.push_event_to_queue(dt=self.dt)
                beaker_2.push_event_to_queue(dt=self.dt)

            # pour the (second) Solute Vessel into the Extraction Vessel
            if do_action == 6:
                # determine the volume to pour from the solvent vessel into the extraction vessel
                d_volume = extract_vessel.get_max_volume() * multiplier

                # push the event to the solvent vessel
                event = ['pour by volume', extract_vessel, d_volume]
                reward = solvent_vessel2.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to either of the beakers
                beaker_1.push_event_to_queue(dt=self.dt)
                beaker_2.push_event_to_queue(dt=self.dt)

            # Indicate that all no more actions are to be completed
            if do_action == 7:
                # pass the fulfilled `done` parameter
                done = True

                # look through each vessel's material dict looking for the target material
                reward = 0

            # redefine the vessels and external_vessels parameters
            vessels = [extract_vessel, beaker_1, beaker_2]
            ext_vessel = [solvent_vessel1, solvent_vessel2]

        return vessels, ext_vessel, reward, done
