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

Module to define the vessel object and it's associated functionality.
:title: vessel.py
:author: Chris Beeler, Nicholas Paquin, and Mitchell Shahen
:history: 2020-07-22
"""

# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-statements
# pylint: disable=unused-argument
# pylint: disable=wrong-import-order

import copy
import math
import numpy as np
import sys
import os
import pickle

sys.path.append("../../")
from chemistrylab.chem_algorithms import material, util
from chemistrylab.extract_algorithms import separate
from chemistrylab.reactions.get_reactions import convert_to_class

# the gas constant (in kPa * L * mol**-1 * K**-1)
R = 8.314462619


class Vessel:
    """
    Class defining the Vessel object.
    """

    def __init__(
            self,
            label,  # Name of the vessel
            temperature=297.0,  # K
            volume=1.0,  # L
            unit='l',
            materials={},  # moles of materials
            solutes={},  # moles of solutes,
            solvents = {},
            v_max=1.0,  # L
            v_min=0.001,  # L
            Tmax=500.0,  # Kelvin
            Tmin=250.0,  # Kelvin
            default_dt=0.05,  # Default time for each step
            n_pixels=100,  # Number of pixel to represent layers
            open_vessel=True,  # True means no lid
            settling_switch=True,  # switch used for defaults
            layer_switch=True,  # switch used for defaults
    ):
        """
        Constructor class for the Vessel object.

        Parameters
        ---------------
        `temperature` : `np.float32`
            The intended temperature of the vessel in Kelvin.
        `volume` : `np.float32`
            The intended vessel volume in units as specified by the `unit` parameter.
        `unit` : `str`
            The intended units of the vessel's volume parameter.
        `materials` : `dict`
            A dictionary containing the vessel's materials, material class representations, and material amounts.
        `solutes` : `dict`
            A dictionary containing the vessel's materials, material class representations, and material amounts.
        `v_max` : `np.float32`
            The maximal volume of the vessel in units as specified by the `unit` parameter.
            Exceeding this value will cause an overflow.
        `v_min` : `np.float32`
            The minimal volume of the vessel in units as specified by the `unit` parameter.
        `Tmax` : `np.float32`
            The maximal temperature of the vessel in Kelvin.
            Exceeding this value will result in the vessel shattering.
        `Tmin` : `np.float32`
            The minimal temperature of the vessel in Kelvin.
        `default_dt` : `np.float32`
            The default time-step of actions occurring in or to the vessel.
        `n_pixels` : `np.int64`
            The number of pixels used to represent layers during extractions.
        `open_vessel` : `bool`
            Whether or not the vessel is open (has no lid).
        `settling_switch` : `bool`
            A switch used for defaults.
        `layer_switch` : `bool`
            A switch used for defaults.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # define the Material Dict and Solute Dict first
        self._material_dict = util.convert_material_dict_units(materials)  # material.name: [material(), amount]; amount is in mole
        self._solute_dict = util.convert_solute_dict_units(solutes)  # solute.name: [solvent(), amount]; amount is in mole
        # initialize parameters
        self.label = label
        self.w2v = None
        self.temperature = temperature
        self.unit = unit
        self.volume = util.convert_volume(volume, unit=self.unit)
        self.pressure = self.get_pressure()
        self.v_max = v_max
        self.v_min = v_min
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.open_vessel = open_vessel
        self.default_dt = default_dt
        self.n_pixels = n_pixels

        # switches
        self.settling_switch = settling_switch
        self.layer_switch = layer_switch

        # create Air （not in material dict)
        self.Air = material.Air()

        # for vessel's pixel representation
        # layer's pixel representation, initially filled with Air's color
        self._layers = np.zeros(self.n_pixels, dtype=np.float32) + self.Air.get_color()

        # layers gaussian representation
        self._layers_position_dict = {self.Air.get_name(): 0.0}  # material.name: position
        for M in self._material_dict:
            # do not fill in solute (solute does not form layer)
            if not self._material_dict[M][0].is_solute():
                self._layers_position_dict[M] = 0.0

        self._layers_variance = 2.0

        # calculate the maximal pressure based on what material is in the vessel
        self.pmax = self.get_pmax()

        # event dict holding all the event functions
        self._event_dict = {
            'temperature change': self._update_temperature,
            'pour by volume': self._pour_by_volume,
            'drain by pixel': self._drain_by_pixel,
            'update material dict': self._update_material_dict,
            'update solute dict': self._update_solute_dict,
            'mix': self._mix,
            'update_layer': self._update_layers,
            'change_heat': self._change_heat,
            'wait': self._wait
        }

        # event queues
        self._event_queue = []  # [['event', parameters], ['event', parameters] ... ]
        self._feedback_queue = []  # [same structure as event queue]
        self.thickness = 1e-3 # beaker is 1mm thick
        self.height_to_diamater = 3/2 # default height to diameter ratio
        self.dimensions = self.get_vessel_dimensions() # returns a tuple (max_height, radius)

    def push_event_to_queue(
            self,
            events=None,  # a list of event tuples(['event', parameters])
            feedback=None,  # a list of event tuples(['event', parameters])
            dt=None,  # time for each step
            check_switches_flag=True  # whether check switches or not
    ):
        """
        Method to assign events to occur by placing them in a queue.

        Parameters
        ---------------
        `events` : `np.float32`
            A list of tuples specifying events to be performed as well as parameters necessary to perform these events.
        `feedback` : `np.float32`
            A list of tuples specifying feedback from events as well as feedback parameters.
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.
        `check_switches_flag` : `bool`
            A switch used for defaults.

        Returns
        ---------------
        `reward` : `np.float64`
            The reward from updating the material dictionary.

        Raises
        ---------------
        None
        """

        # ensure the time-step is specified
        if dt is None:
            dt = self.default_dt

        # queue all events
        if events is not None:
            self._event_queue.extend(events)

        # queue all feedback
        if feedback is not None:
            self._feedback_queue.extend(feedback)

        # call self._update_materials() to execute events / feedback
        reward = self._update_materials(dt, check_switches_flag)

        return reward

    def _update_materials(
            self,
            dt,
            check_switches_flag
    ):
        """
        Method to perform a vessel action.

        Parameters
        ---------------
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.
        `check_switches_flag` : `bool`
            A switch used for defaults.

        Returns
        ---------------
        `reward` : `np.float64`
            The reward from each action in the event queue.

        Raises
        ---------------
        None
        """

        reward = 0

        # loop over event queue until it is empty
        while self._event_queue:
            # get the next event in event_queue
            event = self._event_queue.pop(0)

            # print('-------{}: {} (event)-------'.format(self.label, event[0]))

            # use the name of the event to get the function form event_dict
            action = self._event_dict[event[0]]

            # if action is wait
            if event[0] == 'wait':
                reward += action(self, parameter=event[1:], dt=dt)
            else:
                # call the function corresponding to the event
                reward += action(parameter=event[1:], dt=dt)

        # generate the merged queue from feedback queue and empty the feedback queue
        merged = self._merge_event_queue(
            event_queue=self._feedback_queue,
            dt=dt,
            check_switches_flag=check_switches_flag
        )
        self._feedback_queue = []

        # loop over merged queue until it is empty
        while merged:
            # get the next event in event_queue
            feedback_event = merged.pop(0)

            # print(
            #     '-------{}: {} (feedback_event)-------'.format(
            #         self.label,
            #         feedback_event[0]
            #     )
            # )

            # use the name of the event to get the function form event_dict
            action = self._event_dict[feedback_event[0]]

            # call the function corresponding to the event
            __ = action(parameter=feedback_event[1:], dt=dt)

        return reward

    def _merge_event_queue(
            self,
            event_queue,
            dt,
            check_switches_flag
    ):
        """
        A function to merge events and add default events
        The merge need to be added

        Parameters
        ---------------
        `event_queue` : `list`
            A list containing the events that have been queued.
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.
        `check_switches_flag` : `bool`
            A switch used for defaults.

        Returns
        ---------------
        `merged` : `list`
            The events queue list updated with additional events depending on the switches.

        Raises
        ---------------
        None
        """

        # add default events at the end of merged queue
        merged = copy.deepcopy(event_queue)
        if check_switches_flag:
            if self.settling_switch:
                merged.append(['mix', None])
            if self.layer_switch:
                merged.append(['update_layer'])

            # if any action that needs to be performed or checked each step,
            # it can be added here

        return merged

    # ---------- START EVENT FUNCTIONS ---------- #
    def _wait(self, parameter, dt):
        room_temp = parameter[0]
        initial_temp = self.get_temperature()
        out_beaker = parameter[1]
        wait_until_room = parameter[2]
        if wait_until_room:
            self._update_temperature([room_temp, False], dt)
        else:
            rate = 1.143 * self.get_material_surface_area() * (room_temp - initial_temp)/self.thickness
            heat_change = rate * dt
            self._change_heat([heat_change, out_beaker], dt)
            if initial_temp > room_temp > self.get_temperature() or initial_temp < room_temp < self.get_temperature():
                self._update_temperature([room_temp, False], dt)
        return 0

    def _change_heat(self, parameter, dt):
        """
        Method to modify the temperature and contents of a vessel by adding or removing heat.

        Parameters
        ---------------
        `parameters` : `list`
            A list containing two parameters, the amount of heat available and the intended output beaker.
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.

        Returns
        ---------------
        `0` : `np.int64`
            A default reward of 0.

        Raises
        ---------------
        None
        """

        # extract the inputted parameters
        heat_available = parameter[0] # floating point in joules
        out_beaker = parameter[1] # vessel.Vessel object

        # set up a variable to track the total change of the vessel temperature
        self.total_temp_change = 0
        self.current_temp = self.temperature

        # add logging message for heat change
        # print("Implement Heat Change of {} joules".format(heat_available))

        while heat_available > 0:
            # get the names of the materials in the material dictionary (stored as dict keys)
            material_names = list(self._material_dict.keys())

            # iterate through the material dictionary values for the remaining material parameters
            material_list = []
            for value in list(self._material_dict.values()):
                # we only require the first 2 items (the material class and the material amount)
                necessary_values = value[:2]

                # add the necessary material values to a list for further dissection
                material_list.append(necessary_values)

            # acquire the amounts of each material from the material values list
            material_amounts = [amount for __, amount in material_list]

            # acquire the material objects from the material values list
            material_objs = [material_obj for material_obj, __ in material_list]

            # acquire the boiling points of each material from the material values list
            material_bps = [material_obj()._boiling_point for material_obj in material_objs]

            # ensure all material boiling points are above the current vessel temperature
            try:
                if min(material_bps) < self.temperature:
                    print([(material_obj().get_name(), material_obj()._boiling_point) for material_obj in material_objs])
                    print(min(material_bps))
                    # error for now...
                    exit()
            # if attempting to find the lowest boiling point yields a ValueError (because the boil vessel
            # contains no materials) no further operations will contribute to the distillation of materials
            except ValueError:
                print(
                    "No material remaining in the boil vessel. "
                    "Only the vessel and air will be heated. "
                )

                # use the pressure, volume, and temperature of the vessel
                # to find the molar amount of the air being heated
                air_molar_amount = self.get_pressure() * self.get_volume() / (R * self.get_temperature())

                # acquire the material class representation of Air
                air_mat_class = material.Air

                # calculate the specific heat and entropy and temp change for a vessel of only air (in J/K)
                total_entropy = self._calculate_entropy(
                    material_amounts=[air_molar_amount],
                    material_objs=[air_mat_class]
                )

                # calculate the temperature change of the vessel from the entropy
                temp_change = heat_available / total_entropy

                # set the heat available to 0
                heat_available = 0

                # update the vessel temperature
                self._update_temperature(
                    parameter=[self.temperature + temp_change, False],
                    dt=self.default_dt,
                )

                # updates total temp change and current temp
                self.total_temp_change += self.temperature - self.current_temp
                self.current_temp = self.temperature

                return -1

            # determine the material with the lowest boiling point and its value
            lowest_bp = min(material_bps)

            # find the index of the material with the lowest boiling point
            smallest_bp_index = material_bps.index(lowest_bp)

            # acquire characteristics pertaining to the material with the lowest boiling point
            smallest_bp_name = material_names[smallest_bp_index]
            smallest_bp_amount = material_amounts[smallest_bp_index]
            smallest_bp_obj = material_objs[smallest_bp_index]
            smallest_bp_enth_vap = smallest_bp_obj()._enthalpy_vapor

            # calculate the heat needed to raise the vessel temperature to the lowest boiling point;
            # use Q = mcT, with c = the mass-weighted specific heat capacities of all materials
            temp_change_needed = lowest_bp - self.temperature

            # calculate the total entropy of all the materials in J/K
            total_entropy = self._calculate_entropy(
                material_amounts=material_amounts,
                material_objs=material_objs
            )

            # calculate the energy needed to get to the lowest boiling point
            heat_to_lowest_bp = temp_change_needed * total_entropy

            # if enough heat is available, modify the vessel temp and boil off the material
            if heat_to_lowest_bp < heat_available:
                print("Raising Boil Vessel Temperature by {} Kelvin".format(temp_change_needed))

                # change the vessel temperature to the lowest boiling point
                self._update_temperature(
                    parameter=[lowest_bp, False],
                    dt=self.default_dt
                )

                # updates total temp change and current temp
                self.total_temp_change += self.temperature - self.current_temp
                self.current_temp = self.temperature

                # modify the amount of heat available by subtracting the amount of heat we needed to raise the vessel
                # temperature to the lowest boiling point
                heat_available -= heat_to_lowest_bp

                # calculate the amount of heat needed to boil off all of the material
                heat_to_boil_all = smallest_bp_amount * smallest_bp_enth_vap

                # ensure the beaker has an entry for the material to be boiled off
                if smallest_bp_name not in out_beaker._material_dict.keys():
                    out_beaker._material_dict[smallest_bp_name] = [smallest_bp_obj, 0.0]

                # if enough heat is available, boil off all of the material
                if heat_to_boil_all < heat_available:
                    print("Boiling Off {} mol of {}".format(smallest_bp_amount, smallest_bp_name))

                    # remove the material from the boil vessel's material dictionary
                    del self._material_dict[smallest_bp_name]

                    # add all the material to the beaker's material dictionary
                    out_beaker._material_dict[smallest_bp_name][1] = smallest_bp_amount

                    # modify the heat_change available
                    heat_available -= heat_to_boil_all

                # if not enough heat is available to boil off all the material,
                # boil off enough material to use up all the remaining heat
                else:
                    # calculate the material that is boiled off by using all of the remaining heat
                    boiled_material = heat_available / smallest_bp_enth_vap

                    print("Boiling Off {} mol of {}".format(boiled_material, smallest_bp_name))

                    # subtract the boiled material from the boil vessel's material dictionary
                    self._material_dict[smallest_bp_name][1] -= boiled_material

                    # add the boiled material to the beaker's material dictionary
                    out_beaker._material_dict[smallest_bp_name][1] += boiled_material

                    # reduce the available heat energy to 0
                    heat_available = 0

            # raise the vessel temperature by using all of the available heat energy
            else:
                # calculate the change in vessel temperature if all heat is used
                vessel_temp_change = heat_available / total_entropy

                print("Raising Boil Vessel Temperature by {}".format(vessel_temp_change))

                # modify the boil vessel's temperature accordingly
                self._update_temperature(
                    parameter=[self.temperature + vessel_temp_change, False],
                    dt=self.default_dt
                )

                # updates total temp change and current temp
                self.total_temp_change += self.temperature - self.current_temp
                self.current_temp = self.temperature

                # reduce the available heat energy to 0
                heat_available = 0

        return 0

    def open_lid(self):
        """
        Method to open the vessel.

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

        if self.open_vessel:
            pass
        else:
            self.open_vessel = True
            self.pressure = 1
            # followed by several updates

    def close_lid(self):
        """
        Method to close the vessel.

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

        if not self.open_vessel:
            pass
        else:
            self.open_vessel = False
            # followed by several updates

    def _pour_by_volume(
            self,
            parameter,  # [target_vessel, d_volume, unit] unit is optional, if not included we assume the use of l
            dt
    ):
        """
        Method to pour the contents of the vessel into another vessel.

        Parameters
        ---------------
        `parameter` : `list`
            A list containing 3 elements: the target vessel, the volume of material being moved, and the volume unit.
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.

        Returns
        ---------------
        `reward` : `np.float32`
            The reward associated with the event.

        Raises
        ---------------
        None
        """

        # extract the relevant parameters
        target_vessel = parameter[0]
        d_volume = parameter[1]
        if len(parameter) == 3:
            unit = parameter[2]
        else:
            unit = 'l'
        d_volume = util.convert_volume(d_volume, unit)
        # collect data from the target vessel
        target_material_dict = target_vessel.get_material_dict()
        target_solute_dict = target_vessel.get_solute_dict()
        __, self_total_volume = util.convert_material_dict_and_solute_dict_to_volume(self._material_dict, self._solute_dict, unit="l")

        # set default reward equal to zero
        reward = 0

        # if this vessel is empty or the d_volume is equal to zero, do nothing
        if any([
            math.isclose(self_total_volume, 0.0, rel_tol=1e-6),
            math.isclose(d_volume, 0.0, rel_tol=1e-6)
        ]):
            __ = target_vessel.push_event_to_queue(
                events=None,
                feedback=None,
                dt=dt
            )

            return reward

        # if this vessel has enough volume of material to pour
        if self_total_volume - d_volume > 1e-6:
            # calculate the percentage of amount of materials to be poured
            d_percentage = d_volume / self_total_volume

            for M in copy.deepcopy(self._material_dict):
                # calculate the change of amount(mole) of each material
                d_mole = self._material_dict[M][1] * d_percentage

                # update material dict for this vessel
                self._material_dict[M][1] -= d_mole

                # update material dict for the target vessel, if it's in target vessel
                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                # if it's not in target vessel, create it in material dict
                else:
                    target_material_dict[M] = [
                        copy.deepcopy(self._material_dict[M][0]),
                        d_mole,
                        'mol'
                    ]

            for Solute in copy.deepcopy(self._solute_dict):
                # if solute not in target vessel, create it in solute dict
                if Solute not in target_solute_dict:
                    target_solute_dict[Solute] = {}

                for Solvent in self._solute_dict[Solute]:
                    # calculate the change of amount in each solvent and update the solute dict
                    d_mole = self._solute_dict[Solute][Solvent][1] * d_percentage
                    self._solute_dict[Solute][Solvent][1] -= d_mole

                    # check if the solvent is in target material and update the vessel accordingly
                    if Solvent in target_solute_dict[Solute]:
                        target_solute_dict[Solute][Solvent][1] += d_mole
                    else:
                        target_solute_dict[Solute][Solvent] = [convert_to_class([Solvent])[0], d_mole, 'mol']

        # if this vessel has less liquid than calculated amount then everything gets poured out
        else:
            # update material_dict for both vessels
            for M in copy.deepcopy(self._material_dict):
                # pour all of this material
                d_mole = self._material_dict[M][1]

                # check if this material is in the target vessel and update the material dictionary
                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole, 'mol']

                # remove the material from this vessel
                self._material_dict.pop(M)

                # also remove the material from `self._layers_position_dict` if it's in the dict
                if M in self._layers_position_dict:
                    self._layers_position_dict.pop(M)

            # update solute_dict for both vessels
            for solute in copy.deepcopy(self._solute_dict):
                # if solute not in target vessel, create its entry
                if solute not in target_solute_dict:
                    target_solute_dict[solute] = {}

                for solvent in self._solute_dict[solute]:
                    # calculate the change of amount in each solvent
                    d_mole = self._solute_dict[solute][solvent][1]

                    # check if that solvent is in target material
                    if solvent in target_solute_dict[solute]:
                        target_solute_dict[solute][solvent][1] += d_mole
                    else:
                        target_solute_dict[solute][solvent] = [convert_to_class([solvent])[0], d_mole, 'mol']

                # remove the solute from the solute dictionary
                self._solute_dict.pop(solute)  # pop the solute

        # deal with overflow
        v_max = target_vessel.get_max_volume()
        target_material_dict, target_solute_dict, temp_reward = util.check_overflow(
            material_dict=target_material_dict,
            solute_dict=target_solute_dict,
            v_max=v_max,
            unit=self.unit
        )

        # update the reward
        reward += temp_reward

        # update target vessel's material amount
        event_1 = ['update material dict', target_material_dict]

        # update target vessel's solute dict
        event_2 = ['update solute dict', target_solute_dict]

        # create a event to reset material's position and variance in target vessel
        # push event to target vessel
        __ = target_vessel.push_event_to_queue(
            events=None,
            feedback=[event_1, event_2],
            dt=dt
        )
        __ = target_vessel.push_event_to_queue(
            events=None,
            feedback=None,
            dt=-100000
        )
        return reward

    def _drain_by_pixel(
            self,
            parameter,  # [target_vessel, n_pixel]
            dt
    ):
        """
        Method to drain a vessel gradually by pixel count.

        Parameters
        ---------------
        `parameters` : `list`
            A list containing two elements: the target vessel and the number of pixels to transfer.
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.

        Returns
        ---------------
        `reward` : `list`
            The reward associated with this event.

        Raises
        ---------------
        None
        """

        # extract the necessary parameters
        target_vessel = parameter[0]
        n_pixel = parameter[1]

        # collect data
        target_material_dict = target_vessel.get_material_dict()
        target_solute_dict = target_vessel.get_solute_dict()
        __, self_total_volume = util.convert_material_dict_and_solute_dict_to_volume(self._material_dict, self._solute_dict, unit=self.unit)

        # set default reward equal to zero
        reward = 0

        # if this vessel is empty or the d_volume is equal to zero, do nothing
        if any([
            math.isclose(self_total_volume, 0.0, rel_tol=1e-6),
            math.isclose(n_pixel, 0.0, rel_tol=1e-6)
        ]):
            __ = target_vessel.push_event_to_queue(
                events=None,
                feedback=None,
                dt=dt
            )
            return reward

        # calculate pixels and create a dict to store how many pixels get drained of each color
        color_to_pixel = {}  # float as key may not be a good idea! maybe change this later
        (decimal, integer) = math.modf(n_pixel)  # split decimal and integer
        for i in range(int(integer)):
            color = round(self._layers[i].item(), 3)  # convert to float and round to 3 decimal
            if color not in color_to_pixel:
                color_to_pixel[color] = 1
            else:
                color_to_pixel[color] += 1
        if decimal > 0:
            color = round(self._layers[int(integer)].item(), 3)
            if color not in color_to_pixel:
                color_to_pixel[color] = decimal
            else:
                color_to_pixel[color] += decimal

        # create a dict to calculate the change of each solute for updating the material dict
        d_solute = {}
        for Solute in self._solute_dict:
            d_solute[Solute] = 0.0

        # M is the solvent that's drained out
        for M in copy.deepcopy(self._material_dict):
            # if the solvent's colour is in color_to_pixel (means it get drained)
            if self._material_dict[M][0].get_color() in color_to_pixel:
                # calculate the d_mole of this solvent
                pixel_count = color_to_pixel[self._material_dict[M][0].get_color()]
                d_volume = (pixel_count / self.n_pixels) * self.v_max

                # convert volume to amount (mole)
                density = self._material_dict[M][0].get_density()
                molar_mass = self._material_dict[M][0].get_molar_mass()
                d_mole = util.convert_volume_to_mole(
                    volume=d_volume,
                    density=density,
                    molar_mass=molar_mass
                )

                # if there is some of this solvent left
                if self._material_dict[M][1] - d_mole > 1e-6:
                    # calculate the drained ratio
                    d_percentage = d_mole / self._material_dict[M][1]

                    # update the vessel's material dictionary
                    self._material_dict[M][1] -= d_mole

                    # check if the solvent is in target vessel and update it's material dictionary
                    if M in target_material_dict:
                        target_material_dict[M][1] += d_mole
                    else:
                        target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole, 'mol']

                    # update solute_dict for both vessels
                    for Solute in self._solute_dict:
                        if M in self._solute_dict[Solute]:
                            # calculate the amount of moles
                            d_mole = self._solute_dict[Solute][M][1] * d_percentage

                            # update the material dictionaries
                            self._solute_dict[Solute][M][1] -= d_mole
                            d_solute[Solute] += d_mole

                            # if all solutes are used up, eliminate the solute dictionary
                            if Solute not in target_solute_dict:
                                target_solute_dict[Solute] = {}

                            # update the target solute dictionary
                            if M in target_solute_dict[Solute]:
                                target_solute_dict[Solute][M][1] += d_mole
                            else:
                                target_solute_dict[Solute][M] = [convert_to_class([M])[0], d_mole, 'mol']

                # if all materials are drained out
                else:
                    # calculate the d_mole of this solvent
                    d_mole = self._material_dict[M][1]

                    # check if it's in target vessel and update the material dictionary
                    if M in target_material_dict:
                        target_material_dict[M][1] += d_mole
                    else:
                        target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole, 'mol']

                    self._material_dict.pop(M)
                    self._layers_position_dict.pop(M)

                    # update solute_dict for both vessels
                    for Solute in copy.deepcopy(self._solute_dict):
                        if M in self._solute_dict[Solute]:
                            # calculate the amount to drain
                            d_mole = self._solute_dict[Solute][M][1]

                            # store the change of solute in d_solute
                            d_solute[Solute] += d_mole

                            # if all solutes are used up, eliminate the solute dictionary
                            if Solute not in target_solute_dict:
                                target_solute_dict[Solute] = {}

                            # update the target solute dictionary
                            if M in target_solute_dict[Solute]:
                                target_solute_dict[Solute][M][1] += d_mole
                            else:
                                target_solute_dict[Solute][M] = [convert_to_class([M])[0], d_mole, 'mol']

                            # pop this solvent from the solute
                            self._solute_dict[Solute].pop(M)

        for Solute in copy.deepcopy(self._solute_dict):
            empty_flag = True

            # pop solute from solute dict if it's empty
            if not self._solute_dict[Solute]:
                self._solute_dict.pop(Solute)
                empty_flag = False
            else:
                for Solvent in self._solute_dict[Solute]:
                    if abs(self._solute_dict[Solute][Solvent][1] - 0.0) > 1e-6:
                        empty_flag = False

            if empty_flag:
                self._solute_dict.pop(Solute)

        # use d_solute to update material dict for solute
        for M in d_solute:
            # if M still in solute dict, calculate the amount of it drained
            if M in self._solute_dict:
                d_mole = d_solute[M]
                self._material_dict[M][1] -= d_mole

                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole, 'mol']
            # if M no longer in solute dict, which means all of this solute is drained
            else:
                d_mole = d_solute[M]

                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole, 'mol']

                # pop it from material dict since all drained out
                self._material_dict.pop(M)

        # deal with overflow
        v_max = target_vessel.get_max_volume()
        target_material_dict, target_solute_dict, temp_reward = util.check_overflow(
            material_dict=target_material_dict,
            solute_dict=target_solute_dict,
            v_max=v_max,
            unit=self.unit
        )

        reward += temp_reward

        # print(
        #     '-------{}: {} (new_material_dict)-------'.format(
        #         self.label,
        #         self._material_dict
        #     )
        # )
        # print(
        #     '-------{}: {} (new_solute_dict)-------'.format(
        #         self.label,
        #         self._solute_dict
        #     )
        # )
        # print(
        #     '-------{}: {} (new_material_dict)-------'.format(
        #         target_vessel.label,
        #         target_material_dict
        #     )
        # )
        # print(
        #     '-------{}: {} (new_solute_dict)-------'.format(
        #         target_vessel.label,
        #         target_solute_dict
        #     )
        # )

        # update target vessel's material amount
        event_1 = ['update material dict', target_material_dict]

        # update target vessel's solute dict
        event_2 = ['update solute dict', target_solute_dict]

        # create a event to reset material's position and variance in target vessel

        # push event to target vessel
        __ = target_vessel.push_event_to_queue(
            events=None,
            feedback=[event_1, event_2],
            dt=dt
        )
        __ = target_vessel.push_event_to_queue(
            events=None,
            feedback=None,
            dt=-100000
        )

        return reward

    def _update_material_dict(
            self,
            parameter,  # [new_material_dict]
            dt
    ):
        """
        Method to update the material dictionary.

        Parameters
        ---------------
        `parameters` : `list`
            A list containing a single element, the intended new material dictionary.
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        new_material_dict = util.convert_material_dict_units(parameter[0])

        self._material_dict = util.organize_material_dict(new_material_dict)
        self._layers_position_dict = {self.Air.get_name(): 0.0}  # material.name: position
        for M in self._material_dict:
            # do not fill in solute (solute does not form layer)
            if not self._material_dict[M][0].is_solute():
                self._layers_position_dict[M] = 0.0

    def _update_solute_dict(
            self,
            parameter,  # [new_solute_dict]
            dt
    ):
        """
        Method to update the solute dictionary.

        Parameters
        ---------------
        `parameters` : `list`
            A list containing a single element, the intended new solute dictionary.
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        new_solute_dict = util.convert_solute_dict_units(parameter[0])

        # organize the target_solute_dict so it includes all solute and solvent
        self._solute_dict = util.organize_solute_dict(
            material_dict=self._material_dict,
            solute_dict=new_solute_dict
        )

    def _mix(
            self,
            parameter,  # [mixing_parameter]
            dt
    ):
        """
        Method to mix a vessel.

        Parameters
        ---------------
        `parameters` : `list`
            A list containing a single element, mixing parameters.
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.

        Returns
        ---------------
        `reward` : `np.float32`
            The reward associated with this event.

        Raises
        ---------------
        None
        """
        # set default reward equal to zero
        reward = 0

        # get mixing parameter from parameter if mixing, or get dt as mixing parameter for settling
        mixing_parameter = parameter[0] if parameter[0] else dt

        # collect data (according to separate.mix()):

        # convert the material dictionary to volumes in mL
        self_volume_dict, __ = util.convert_material_dict_and_solute_dict_to_volume(self._material_dict, self._solute_dict, unit=self.unit)

        layers_variance = self._layers_variance

        solvent_volume = []
        layers_position = []
        layers_density = []
        solute_polarity = []
        solvent_polarity = []
        solute_amount = []

        if self._solute_dict:
            for Solute in self._solute_dict:
                # append empty lists for each solute
                solute_amount.append([])

                # collect polarity
                solute_polarity.append(self._material_dict[Solute][0].get_polarity())
        else:
            solute_amount.append([])
            solute_polarity.append(0.0)

        for M in self._layers_position_dict:  # if M forms a layer
            if M == 'Air':  # skip Air for now
                continue

            # if material exists in dictionary
            if M in self._material_dict.keys():
                # if the material is a solvent
                if self._material_dict[M][0].is_solvent():
                    solvent_volume.append(self_volume_dict[M])
                    solvent_polarity.append(self._material_dict[M][0].get_polarity())
                    solute_counter = 0

                    if self._solute_dict:
                        # fill in solute_amount
                        for Solute in self._solute_dict:
                            solute_amount[solute_counter].append(self._solute_dict[Solute][M][1])
                            solute_counter += 1
                    else:
                        solute_amount[0].append(0.0)

            # if material exists in dictionary
            if M in self._material_dict.keys():
                layers_position.append(self._layers_position_dict[M])
                layers_density.append(self._material_dict[M][0].get_density())

        # Add Air
        layers_position.append(self._layers_position_dict['Air'])
        layers_density.append(self.Air.get_density())

        # execute the mix function
        new_layers_position, self._layers_variance, new_solute_amount, __ = separate.mix(
            A=np.array(solvent_volume),
            B=np.array(layers_position),
            C=layers_variance,
            D=np.array(layers_density),
            Spol=np.array(solute_polarity),
            Lpol=np.array(solvent_polarity),
            S=np.array(solute_amount),
            mixing=mixing_parameter
        )

        # use results returned from 'mix' to update self._layers_position_dict and self._solute_dict
        layers_counter = 0  # count position for layers in new_layers_position
        solvent_counter = 0  # count position for solvent in new_solute_amount for each solute
        for M in self._layers_position_dict:
            if M == 'Air':  # skip Air for now
                continue
            self._layers_position_dict[M] = new_layers_position[layers_counter]
            if self._solute_dict:
                layers_counter += 1
                if self._material_dict[M][0].is_solvent():
                    solute_counter = 0  # count position for solute in new_solute_amount
                    for Solute in self._solute_dict:
                        solute_amount = new_solute_amount[solute_counter][solvent_counter]
                        self._solute_dict[Solute][M][1] = solute_amount
                        solute_counter += 1
                    solvent_counter += 1

        # deal with Air
        self._layers_position_dict['Air'] = new_layers_position[layers_counter]

        return reward

    def _update_layers(
            self,
            parameter,
            dt
    ):
        """
        Modify self._layers.

        Parameters
        ---------------
        `parameters` : `list`
            A list containing parameters to properly perform this events.
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # collect data (according to separate.map())
        layers_amount = []
        layers_position = []
        layers_color = []
        layers_variance = self._layers_variance
        self_volume_dict, self_total_volume = util.convert_material_dict_and_solute_dict_to_volume(
            self._material_dict,
            solute_dict=self._solute_dict,
            unit=self.unit
        )

        for M in self._layers_position_dict:
            if M == 'Air':
                continue

            # if material exists in dictionary
            if M in self._material_dict.keys():
                # if not a solute
                if not self._material_dict[M][0].is_solute():
                    layers_amount.append(self_volume_dict[M])
                    layers_position.append((self._layers_position_dict[M]))
                    layers_color.append(self._material_dict[M][0].get_color())

        # calculate air
        air_volume = self.v_max - self_total_volume
        if air_volume < 1e-6:
            air_volume = 0.0
        layers_amount.append(air_volume)
        layers_position.append(self._layers_position_dict['Air'])
        layers_color.append(self.Air.get_color())

        self._layers = separate.map_to_state(
            A=np.array(layers_amount),
            B=np.array(layers_position),
            C=layers_variance,
            colors=layers_color,
            x=separate.x
        )

    # function to set the volume of the container and to specify the units
    def set_volume(self, volume: float, unit='l', override=False):
        """
        Method to set the volume of the vessel and specify the units of the volume of the vessel.

        Parameters
        ---------------
        `volume` : `np.float32`
            The intended new volume of the vessel.
        `unit` : `str`
            The units of the volume of the vessel.
        `override` : `bool`
            Whether or not the material and solute dictionaries are NOT overwritten in the event of an overflow.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        if not self._material_dict and not self._solute_dict:
            self.volume = util.convert_volume(volume, unit)
        elif override:
            self.volume = util.convert_volume(volume, unit)
            self._material_dict, self._solute_dict,  _ = util.check_overflow(self._material_dict, self._solute_dict,
                                                                            self.volume, unit)
        else:
            raise ValueError('Material dictionary or solute dictionary is not empty')

    def set_v_min(self, volume: float, unit='l'):
        """
        Method to set the minimal volume of the vessel and specify the units of the volume of the vessel.

        Parameters
        ---------------
        `volume` : `np.float32`
            The intended new volume of the vessel.
        `unit` : `str`
            The units of the volume of the vessel.
        `override` : `bool`
            Whether or not the material and solute dictionaries are NOT overwritten in the event of an overflow.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        self.v_min = util.convert_volume(volume, unit)

    def set_v_max(self, volume: float, unit='l'):
        """
        Method to set the maximal volume of the vessel and specify the units of the volume of the vessel.

        Parameters
        ---------------
        `volume` : `np.float32`
            The intended new volume of the vessel.
        `unit` : `str`
            The units of the volume of the vessel.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        self.v_max = util.convert_volume(volume, unit)

    def _update_temperature(
            self,
            parameter,  # [target_temperature, override]
            dt
    ):
        """
        Method to update the temperature of the vessel.

        Parameters
        ---------------
        `parameters` : `list`
            A list containing two elements, the target temperature and
            a boolean indicating if the bounds are to be overridden.
        `dt` : `np.float32`
            The time-step of actions occurring in or to the vessel.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        feedback = []

        # unpack the provided parameters
        target_temperature = parameter[0]
        override = parameter[1]

        if not override:
            # check that the target temperature is within the bounds set by the vessel's temperature constraints
            if target_temperature > self.Tmax:
                target_temperature = self.Tmax
                print("Target temperature in not within the temperature bounds imposed on the vessel.")
            elif target_temperature < self.Tmin:
                target_temperature = self.Tmin
                print("Target temperature in not within the temperature bounds imposed on the vessel.")

            # set the temperature change
            self.temperature = target_temperature

        else:
            # reset the temperature constraints using the target temperature
            if target_temperature > self.Tmax:
                self.Tmax = target_temperature
            elif target_temperature < self.Tmin:
                self.Tmin = target_temperature

            # update vessel's temperature property
            self.temperature = target_temperature

        # loop over all the materials in the vessel obtaining feedback from updating the temperatures of each material
        for material_obj in self._material_dict.values():
            feedback = material_obj[0]().update_temperature(
                target_temperature=target_temperature,
                dt=dt
            )
            self._feedback_queue.extend(feedback)

    @staticmethod
    def _calculate_entropy(material_amounts, material_objs):
        """
        Method to calculate the entropy of a vessel containing the materials listed in the materials
        parameter. Also required are the material classes.

        Parameters
        ---------------
        `material_amounts` : `list`
            A list containing the amounts of each material in the vessel.
        `material_objs` : `np.float32`
            A list containing the class representations of the materials in the vessel.

        Returns
        ---------------
        `0` : `np.int64`
            A default reward of 0.

        Raises
        ---------------
        None
        """

        # calculate the total entropy of all the materials in J/K
        total_entropy = 0
        for i, material_amount in enumerate(material_amounts):
            specific_heat = material_objs[i]()._specific_heat  # in J/g*K
            molar_amount = material_amount  # in mol
            molar_mass = material_objs[i]()._molar_mass  # in g/mol

            # calculate the entropy
            material_entropy = specific_heat * molar_amount * molar_mass
            total_entropy += material_entropy

        return total_entropy

    def get_concentration(self, materials=[]):
        """
        Method to convert molar volume to concentration.

        Parameters
        ---------------
        `materials` : `list`
            List of materials to check against the material dictionary.

        Returns
        ---------------
        `C` : `np.array`
            An array of the concentrations (in mol/L) of each chemical in the experiment.

        Raises
        ---------------
        None
        """

        # if not initially provided, fill the materials list with the materials in the vessel
        if not materials:
            materials = list(self._material_dict.keys())

        # get the vessel volume
        V = self.get_current_volume()[-1]

        # set up lists to contain the vessel's materials and the material amounts
        materials_in_vessel = []
        n_list = []

        # iteracte through the material dictionary to update the above lists
        for key, item in self._material_dict.items():
            materials_in_vessel.append(key)
            n_list.append(item[1])

        # convert the list of amounts to an array
        n = np.array(n_list)

        # create an array containing the concentrations of each chemical
        C = np.zeros(len(materials), dtype=np.float32)

        # iterate through the materials required in the concentration array
        for i, req_material in enumerate(materials):
            # if the requested material is not in the vessel, assign a concentration of 0
            if req_material not in materials_in_vessel:
                C[i] = 0
            else:
                # if the requested material is found, acquire the material's concentration in the vessel
                material_index = materials_in_vessel.index(req_material)
                C[i] = n[material_index] / V

        return C

    # functions to access private properties
    def get_material_amount(
            self,
            material_name=None
    ):
        """
        Method to get the material amount of a material in the material dictionary.

        Parameters
        ---------------
        `material_name` : `str`
            The name of the material in the material dictionary.

        Returns
        ---------------
        `amount` : `np.float32`
            The amount of the requested material.

        Raises
        ---------------
        None
        """

        if isinstance(material_name, list):
            amount = []
            for i in material_name:
                try:
                    amount.append(self._material_dict[i][1])
                except KeyError:
                    amount.append(0.0)
        else:
            try:
                amount = self._material_dict[material_name][1]
            except KeyError:
                amount = 0.0

        return amount

    def get_material_volume(
            self,
            material_name=None
    ):
        """
        Method to obtain the volume of a material in the vessel.

        Parameters
        ---------------
        `material_name` : `str`
            The name of the material in the material dictionary.

        Returns
        ---------------
        `amount` : `np.float32`
            The volumetric amount of the material.

        Raises
        ---------------
        None
        """

        self_volume_dict, self_total_volume = util.convert_material_dict_and_solute_dict_to_volume(
            self._material_dict,
            self._solute_dict,
            unit=self.unit
        )

        if isinstance(material_name, list):
            amount = []
            for i in material_name:
                try:
                    amount.append(self_volume_dict[i])
                except KeyError:
                    if material_name == 'Air':
                        return self.v_max - self_total_volume
                    amount.append(0.0)
        else:
            try:
                amount = self_volume_dict[material_name]
            except KeyError:
                if material_name == 'Air':
                    amount = self.v_max - self_total_volume
                else:
                    amount = 0.0

        return amount

    def get_material_position(
            self,
            material_name=None
    ):
        """
        Method to get the position of the material in a layer.

        Parameters
        ---------------
        `material_name` : `str`
            The name of the material in the material dictionary.

        Returns
        ---------------
        `position` : `np.int64`
            The position of the material in the layer dictionary.

        Raises
        ---------------
        None
        """

        if isinstance(material_name, list):
            position = []
            for i in material_name:
                try:
                    position.append(self._layers_position_dict[i])
                except KeyError:
                    position.append(0.0)
                    if i != 'Air':
                        print("Error: requesting position for {} (non-existing material)".format(i))
        else:
            try:
                position = self._layers_position_dict[material_name]
            except KeyError:
                if material_name != 'Air':
                    print(
                        "Error: requesting position for {} (non-existing material)".format(
                            material_name
                        )
                    )
                position = 0.0

        return position

    def get_material_variance(self):
        """
        Method to get the material variance property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `layers_variance` : `np.float32`
            The variance of the layers dictionary.

        Raises
        ---------------
        None
        """

        return self._layers_variance

    def get_layers(self):
        """
        Method to get the vessel layers property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `amount` : `np.array`
            The array containing layers.

        Raises
        ---------------
        None
        """

        return np.copy(self._layers)

    def get_material_color(self, material_name):
        """
        Method to get the material color property.

        Parameters
        ---------------
        `material_name` : `np.float32`
            The name of the material in the material dictionary.

        Returns
        ---------------
        `material_color` : `np.float32`
            A number representing the color of the specified material.

        Raises
        ---------------
        None
        """

        return self._material_dict[material_name][0].get_color()

    def get_material_dict(self):
        """
        Method to get the material dictionary property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `material_dict` : `dict`
            The vessel's material dictionary.

        Raises
        ---------------
        None
        """

        return copy.deepcopy(self._material_dict)

    def get_solute_dict(self):
        """
        Method to get the solute dictionary property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `solute_dict` : `dict`
            The vessel's solute dictionary.

        Raises
        ---------------
        None
        """

        return copy.deepcopy(self._solute_dict)

    def get_position_and_variance(
            self,
            dict_or_list='dict'
    ):
        """
        Method to get the material position and variance property.

        Parameters
        ---------------
        `dict_or_list` : `str`
            Indicating if the output position list should be a list or dict.

        Returns
        ---------------
        `position_list` : `list` or `dict`
            An object containing the layer positions as a list or a dict.
        `layers_variance` : `np.float64`
            The variance of the layers dictionary.

        Raises
        ---------------
        None
        """

        position_list = []

        if dict_or_list == 'dict':
            position_list = copy.deepcopy(self._layers_position_dict)
        elif dict_or_list == 'list':
            for L in self._layers_position_dict:
                position_list.append(self._layers_position_dict[L])

        return position_list, self._layers_variance

    def get_max_volume(self):
        """
        Method to get the vessel's maximal volume property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `v_max` : `np.float32`
            The maximal volume of the vessel.

        Raises
        ---------------
        None
        """

        return self.v_max

    def get_min_volume(self):
        """
        Method to get the vessel's minimal volume property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `v_min` : `np.float32`
            The minimum volume of the vessel.

        Raises
        ---------------
        None
        """

        return self.v_min

    def get_Tmin(self):
        """
        Method to get the vessel's minimal temperature property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `Tmin` : `np.float32`
            The minimum temperature of the vessel.

        Raises
        ---------------
        None
        """

        return self.Tmin

    def get_Tmax(self):
        """
        Method to get the vessel's maximal temperature property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `Tmax` : `np.float32`
            The maximal temperature of the vessel.

        Raises
        ---------------
        None
        """

        return self.Tmax

    def get_pmax(self):
        """
        Method to get the vessel's maximal pressure property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `max_pressure` : `np.float32`
            The maximal pressure of the vessel based on the vessel's material dictionary,
            maximal temperature, and minimal volume.

        Raises
        ---------------
        None
        """

        # set up a variable to contain the total pressure
        max_pressure = 0

        # calculate the total pressure in a vessel using the material dictionary
        for item in self._material_dict.items():
            max_pressure += item[1][1] * R * self.Tmax / self.v_min

        # if the vessel contains no material use 1 atm as a baseline (in kPa)
        max_pressure = 101.325

        return max_pressure

    def get_pressure(self):
        """
        Method to get the vessel's maximal pressure property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `total_pressure` : `np.float32`
            The total pressure of the vessel from aggregate pressures based on the vessel's material dictionary,
            maximal temperature, and minimal volume.

        Raises
        ---------------
        None
        """

        # set up a variable to contain the total pressure
        total_pressure = 0

        # calculate the total pressure in a vessel using the material dictionary
        for item in self._material_dict.items():
            total_pressure += item[1][1] * R * self.temperature / self.volume

        # if the vessel contains no material use 1 atm as a baseline (in kPa)
        if total_pressure == 0:
            total_pressure = 101.325

        return total_pressure

    def get_part_pressure(self):
        """
        Method to get the vessel's maximal pressure property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `part_pressures` : `list`
            A list of the partial pressures of each of the materials in the vessel.

        Raises
        ---------------
        None
        """

        # set up a variable to contain the partial pressures
        part_pressures = np.zeros(len(self._material_dict))

        # calculate the total pressure in a vessel using the material dictionary
        for i, item in enumerate(self._material_dict.items()):
            part_pressures[i] = item[1][1] * R * self.temperature / self.volume

        return part_pressures

    def get_defaultdt(self):
        """
        Method to get the vessel's default time-step property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `default_dt` : `np.float32`
            The default time-step of the vessel.

        Raises
        ---------------
        None
        """

        return self.default_dt

    def get_temperature(self):
        """
        Method to get the vessel's temperature property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `temperature` : `np.float32`
            The vessel temperature.

        Raises
        ---------------
        None
        """

        return self.temperature

    def get_volume(self):
        """
        Method to get the vessel's volume property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `volume` : `np.float32`
            The vessel volume.

        Raises
        ---------------
        None
        """

        return self.volume

    def get_current_volume(self):
        """
        Method to get the vessel's current volume property.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `volume` : `np.float32`
            The volume of the materials in the material dictionary in the vessel's volume.

        Raises
        ---------------
        None
        """

        return util.convert_material_dict_and_solute_dict_to_volume(self._material_dict, self._solute_dict, unit=self.unit)

    def get_vessel_dimensions(self):
        vol = self.v_max
        diam = (vol / np.pi * 4 / self.height_to_diamater) ** 1 / 3
        height = diam * self.height_to_diamater
        return height, diam / 2

    def get_material_surface_area(self):
        vol = self.get_current_volume()[-1]
        height = vol / np.pi / (self.dimensions[-1] ** 2)
        surface_area = 2 * np.pi * self.dimensions[-1] ** 2 + 2 * np.pi * self.dimensions[-1] * height
        return surface_area

    def define_height_to_diameter(self, ratio):
        self.height_to_diamater = ratio


    # ---------- SAVING/LOADING FUNCTIONS ---------- #

    def save_vessel(self, vessel_rootname: str):
        """
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
        """

        # use the vessel path that was given upon initialization or during parameter verification
        output_directory = os.path.dirname(os.path.realpath('reaction_bench_v0_engine.py'))
        vessel_filename = "{}.pickle".format(vessel_rootname)
        open_file = os.path.join(output_directory, vessel_filename)

        # delete any existing vessel files to ensure the vessel is saved as intended
        if os.path.exists(open_file):
            os.remove(open_file)

        # open the intended vessel file and save the vessel as a pickle file
        with open(open_file, 'wb') as vessel_file:
            pickle.dump(self, vessel_file)

    def load_vessel(self, vessel_path: str):
        """
        Method to load a vessel from a pickle file.

        Parameters
        ---------------
        `vessel_path` : `str` (default="")
            The filepath to the pickle file containing the vessel (no extension; not the absolute path).

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        with open(vessel_path, 'rb') as handle:
            v = pickle.load(handle)
            self._material_dict = v._material_dict  # material.name: [material(), amount]; amount is in mole
            self._solute_dict = v._solute_dict  # solute.name: [solvent(), amount]; amount is in mole

            # initialize parameters
            self.label = v.label
            self.w2v = v.w2v
            self.temperature = v.temperature
            self.pressure = v.pressure
            self.volume = v.volume
            self.v_max = v.v_max
            self.v_min = v.v_min
            self.Tmax = v.Tmax
            self.Tmin = v.Tmin
            self.open_vessel = v.open_vessel
            self.default_dt = v.default_dt
            self.n_pixels = v.n_pixels

            # switches
            self.settling_switch = v.settling_switch
            self.layer_switch = v.layer_switch

            # create Air （not in material dict)
            self.Air = v.Air

            # for vessel's pixel representation
            # layer's pixel representation, initially filled with Air's color
            self._layers = v._layers

            # layers gaussian representation
            self._layers_position_dict = v._layers_position_dict
            self._layers_variance = v._layers_variance

            # calculate the maximal pressure based on what material is in the vessel
            self.pmax = v.pmax

            # event dict holding all the event functions
            self._event_dict = {
                'temperature change': self._update_temperature,
                'pour by volume': self._pour_by_volume,
                'drain by pixel': self._drain_by_pixel,
                'update material dict': self._update_material_dict,
                'update solute dict': self._update_solute_dict,
                'mix': self._mix,
                'update_layer': self._update_layers,
                'change_heat': self._change_heat,
                'wait': self._wait
            }

            # event queues
            self._event_queue = v._event_queue  # [['event', parameters], ['event', parameters] ... ]
            self._feedback_queue = v._feedback_queue  # [same structure as event queue]
