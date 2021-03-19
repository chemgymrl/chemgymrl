'''
Module to define the vessel object and it's associated functionality.
:title: vessel.py
:author: Chris Beeler and Mitchell Shahen
:history: 2020-07-22
'''

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

sys.path.append("../../")
from chemistrylab.chem_algorithms import material, util
from chemistrylab.extract_algorithms import separate

# the gas constant (in kPa * L * mol**-1 * K**-1)
R = 8.314462619

class Vessel:
    '''
    Class defining the Vessel object.
    '''

    def __init__(
            self,
            label, # Name of the vessel
            temperature=297, # K
            volume=1.0, # L
            materials={}, # moles of materials
            solutes={}, # moles of solutes
            v_max=1.0, # L
            v_min=0.001, # L
            Tmax=500.0, # Kelvin
            Tmin=250.0, # Kelvin
            default_dt=0.05, # Default time for each step
            n_pixels=100, # Number of pixel to represent layers
            open_vessel=True, # True means no lid
            settling_switch=True, # switch used for defaults
            layer_switch=True, # switch used for defaults
    ):
        '''
        Constructor class for the Vessel object.
        '''

        # define the Material Dict and Solute Dict first
        self._material_dict = materials  # material.name: [material(), amount]; amount is in mole
        self._solute_dict = solutes  # solute.name: [solvent(), amount]; amount is in mole

        # initialize parameters
        self.label = label
        self.w2v = None
        self.temperature = temperature
        self.pressure = self.get_pressure()
        self.volume = volume
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
        self._layers_variance = 2.0

        # calculate the maximal pressure based on what material is in the vessel
        self.pmax = self.get_pmax()

        # event dict holding all the event functions
        self._event_dict = {
            'temperature change': self._update_temperature,
            'pour by volume': self._pour_by_volume,
            'drain by pixel': self._drain_by_pixel,
            'fully mix': self._fully_mix,
            'update material dict': self._update_material_dict,
            'update solute dict': self._update_solute_dict,
            'mix': self._mix,
            'update_layer': self._update_layers,
            'change_heat': self._change_heat
        }

        # event queues
        self._event_queue = []  # [['event', parameters], ['event', parameters] ... ]
        self._feedback_queue = []  # [same structure as event queue]

    def push_event_to_queue(
            self,
            events=None, # a list of event tuples(['event', parameters])
            feedback=None, # a list of event tuples(['event', parameters])
            dt=None, # time for each step
            check_switches_flag=True # whether check switches or not
    ):
        '''
        Method to assign events to occur by placing them in a queue.
        '''

        # set the default reward to 0
        reward = 0

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
        '''
        Method to perform a vessel action.
        '''

        reward = 0

        # loop over event queue until it is empty
        while self._event_queue:
            # get the next event in event_queue
            event = self._event_queue.pop(0)

            print('-------{}: {} (event)-------'.format(self.label, event[0]))

            # use the name of the event to get the function form event_dict
            action = self._event_dict[event[0]]

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

            print(
                '-------{}: {} (feedback_event)-------'.format(
                    self.label,
                    feedback_event[0]
                )
            )

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
        '''
        A function to merge events and add default events
        The merge need to be added
        '''

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

    ## ---------- START EVENT FUNCTIONS ---------- ##

    def _update_temperature(
            self,
            parameters,  # [target_temperature]
            dt
    ):
        '''
        Method to update the temperature of the vessel.
        '''

        feedback = []

        # update vessel's temperature property
        self.temperature += parameters[0]

        # loop over all the materials in the vessel
        for material_obj in self._material_dict.values():
            feedback = material_obj[0].update_temperature(
                target_temperature=parameters[0],
                dt=dt
            )
            self._feedback_queue.extend(feedback)

    def _change_heat(self, parameter, dt):
        '''
        Method to modify the temperature and contents of a vessel by adding or removing heat.
        '''

        # extract the inputted parameters
        heat_available = parameter[0]
        out_beaker = parameter[1]

        # add logging message for heat change
        # print("Implement Heat Change of {} joules".format(heat_available))

        while heat_available > 0:
            # get the necessary vessel properties
            material_names = list(self._material_dict.keys())
            material_amounts = [amount for __, amount in self._material_dict.values()]
            material_objs = [material_obj for material_obj, __ in self._material_dict.values()]
            material_bps = [material_obj()._boiling_point for material_obj in material_objs]
            material_sp_heats = [material_obj()._specific_heat for material_obj in material_objs]

            # ensure all material boiling points are above the current vessel temperature
            if len(material_bps) == 0:
                raise ValueError("No material remaining in the boil vessel.")
            # ensure all material boiling points are above the current vessel temperature
            else:
                if min(material_bps) < self.temperature:
                    raise IOError(
                        "An error has caused the temperature of the vessel "
                        "to exceed the boiling point of at least one material."
                    )

            # determine the material with the smallest boiling point and its value
            smallest_bp = min(material_bps)

            # get characteristics pertaining to the material with the smallest boiling point;
            # this is the material to watch out for during the current iteration
            smallest_bp_index = material_bps.index(smallest_bp)
            smallest_bp_name = material_names[smallest_bp_index]
            smallest_bp_amount = material_amounts[smallest_bp_index]
            smallest_bp_obj = material_objs[smallest_bp_index]
            smallest_bp_enth_vap = smallest_bp_obj()._enthalpy_vapor

            # calculate the heat needed to raise the vessel temperature to the smallest boiling point;
            # use Q = mcT, with c = the mass-weighted specific heat capacities of all materials
            temp_change_needed = smallest_bp - self.temperature

            # calculate the total entropy of all the materials (in J/K)
            total_entropy = 0
            for i, material_amount in enumerate(material_amounts):
                specific_heat = material_sp_heats[i] # in J/g*K
                molar_amount = material_amount # in mol
                molar_mass = material_objs[i]()._molar_mass # in g/mol

                # calculate the entropy
                material_entropy = specific_heat * molar_amount * molar_mass
                total_entropy += material_entropy

            # calculate the energy needed to get to the smallest boiling point
            heat_to_add = temp_change_needed * total_entropy

            # if enough heat is available, change the vessel's temperature
            # immediately to the lowest boiling point of the materials and
            # use the remaining heat energy to boil off the material
            if heat_to_add < heat_available:
                # add logging message
                # print("Raising Boil Vessel Temperature by {} Kelvin".format(temp_change_needed))

                # change the vessel temperature to the smallest boiling point
                self.temperature = smallest_bp

                # modify the amount of heat available
                heat_available -= heat_to_add

                # calculate the amount of heat needed to boil off all of the material
                heat_to_boil_all = smallest_bp_amount * smallest_bp_enth_vap

                # ensure the beaker to contain the boiled off
                # material has an entry for said material
                if smallest_bp_name not in out_beaker._material_dict.keys():
                    out_beaker._material_dict[smallest_bp_name] = [smallest_bp_obj, 0.0]

                # if enough heat is available, boil off all of the material by deleting it from the
                # boil vessel's material dictionary and adding all the material to the beaker
                if heat_to_boil_all < heat_available:
                    # add a logging message
                    # print("Boiling Off {} mol of {}".format(smallest_bp_amount, smallest_bp_name))

                    # remove the material from the boil vessel's material dictionary
                    del self._material_dict[smallest_bp_name]

                    # add all the material to the beaker's material dictionary
                    out_beaker._material_dict[smallest_bp_name][1] = smallest_bp_amount

                    # modify the heat_change available to continue the while loop
                    heat_available -= heat_to_boil_all

                # if not enough heat is available to boil off all the material,
                # boil off enough material to use up all the remaining heat
                else:
                    # calculate the material that is boiled off by using all of the remaining heat
                    boiled_material = heat_available / smallest_bp_enth_vap

                    # add a logging message
                    # print("Boiling Off {} mol of {}".format(boiled_material, smallest_bp_name))

                    # subtract the boiled material from the boil vessel's material dictionary
                    self._material_dict[smallest_bp_name][1] -= boiled_material

                    # add the boiled material to the beaker's material dictionary
                    out_beaker._material_dict[smallest_bp_name][1] += boiled_material

                    # reduce the available heat energy to 0
                    heat_available = 0

            # if the amount of heat energy is not sufficient to reach the lowest available boiling
            # point, raise the vessel temperature by using all of the available heat energy
            else:
                # calculate the change in vessel temperature if all heat is used
                vessel_temp_change = heat_available / total_entropy

                # add logging message
                # print("Raising Boil Vessel Temperature by {}".format(vessel_temp_change))

                # modify the boil vessel's temperature accordingly
                self.temperature += vessel_temp_change

                # reduce the available heat energy to 0
                heat_available = 0

        return 0

    def open_lid(self):
        '''
        Method to open the vessel.
        '''

        if self.open_vessel:
            pass
        else:
            self.open_vessel = True
            self.pressure = 1
            # followed by several updates

    def close_lid(self):
        '''
        Method to close the vessel.
        '''

        if not self.open_vessel:
            pass
        else:
            self.open_vessel = False
            # followed by several updates

    def _pour_by_volume(
            self,
            parameter,  # [target_vessel, d_volume]
            dt
    ):
        '''
        Method to pour the contents of the vessel into another vessel.
        '''

        # extract the relevant parameters
        target_vessel = parameter[0]
        d_volume = parameter[1]

        # collect data from the target vessel
        target_material_dict = target_vessel.get_material_dict()
        target_solute_dict = target_vessel.get_solute_dict()
        __, self_total_volume = util.convert_material_dict_to_volume(self._material_dict)

        # set the default reward to 0
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
                        d_mole
                    ]

            for Solute in copy.deepcopy(self._solute_dict):
                # if solute not in target vessel, create it in solute dict
                if Solute not in target_solute_dict:
                    target_solute_dict[Solute] = {}

                for Solvent in self._solute_dict[Solute]:
                    # calculate the change of amount in each solvent and update the solute dict
                    d_mole = self._solute_dict[Solute][Solvent] * d_percentage
                    self._solute_dict[Solute][Solvent] -= d_mole

                    # check if the solvent is in target material and update the vessel accordingly
                    if Solvent in target_solute_dict[Solute]:
                        target_solute_dict[Solute][Solvent] += d_mole
                    else:
                        target_solute_dict[Solute][Solvent] = d_mole

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
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]

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
                    d_mole = self._solute_dict[solute][solvent]

                    # check if that solvent is in target material
                    if solvent in target_solute_dict[solute]:
                        target_solute_dict[solute][solvent] += d_mole
                    else:
                        target_solute_dict[solute][solvent] = d_mole

                # remove the solute from the solute dictionary
                self._solute_dict.pop(solute)  # pop the solute

        # deal with overflow
        v_max = target_vessel.get_max_volume()
        target_material_dict, target_solute_dict, temp_reward = util.check_overflow(
            material_dict=target_material_dict,
            solute_dict=target_solute_dict,
            v_max=v_max
        )

        # update the reward
        reward += temp_reward

        # update target vessel's material amount
        event_1 = ['update material dict', target_material_dict]

        # update target vessel's solute dict
        event_2 = ['update solute dict', target_solute_dict]

        # create a event to reset material's position and variance in target vessel
        event_3 = ['fully mix']

        # push event to target vessel
        __ = target_vessel.push_event_to_queue(
            events=None,
            feedback=[event_1, event_2, event_3],
            dt=dt
        )

        return reward

    def _drain_by_pixel(
            self,
            parameter,  # [target_vessel, n_pixel]
            dt
    ):
        '''
        Method to drain a vessel gradually by pixel count.
        '''

        # extract the necessary parameters
        target_vessel = parameter[0]
        n_pixel = parameter[1]

        # collect data
        target_material_dict = target_vessel.get_material_dict()
        target_solute_dict = target_vessel.get_solute_dict()
        __, self_total_volume = util.convert_material_dict_to_volume(self._material_dict)

        # print the old and new dictionaries to the terminal
        print(
            '-------{}: {} (old_material_dict)-------'.format(
                self.label,
                self._material_dict
            )
        )
        print(
            '-------{}: {} (old_solute_dict)-------'.format(
                self.label,
                self._solute_dict
            )
        )
        print(
            '-------{}: {} (old_material_dict)-------'.format(
                target_vessel.label,
                target_material_dict
            )
        )
        print(
            '-------{}: {} (old_solute_dict)-------'.format(
                target_vessel.label,
                target_solute_dict
            )
        )

        # set the default reward to 0
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
            if self._material_dict[M][0]().get_color() in color_to_pixel:
                # calculate the d_mole of this solvent
                pixel_count = color_to_pixel[self._material_dict[M][0]().get_color()]
                d_volume = (pixel_count / self.n_pixels) * self.v_max

                # convert volume to amount (mole)
                density = self._material_dict[M][0]().get_density()
                molar_mass = self._material_dict[M][0]().get_molar_mass()
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
                        target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]

                    # update solute_dict for both vessels
                    for Solute in self._solute_dict:
                        if M in self._solute_dict[Solute]:
                            # calculate the amount of moles
                            d_mole = self._solute_dict[Solute][M] * d_percentage

                            # update the material dictionaries
                            self._solute_dict[Solute][M] -= d_mole
                            d_solute[Solute] += d_mole

                            # if all solutes are used up, eliminate the solute dictionary
                            if Solute not in target_solute_dict:
                                target_solute_dict[Solute] = {}

                            # update the target solute dictionary
                            if M in target_solute_dict[Solute]:
                                target_solute_dict[Solute][M] += d_mole
                            else:
                                target_solute_dict[Solute][M] = d_mole

                # if all materials are drained out
                else:
                    # calculate the d_mole of this solvent
                    d_mole = self._material_dict[M][1]

                    # check if it's in target vessel and update the material dictionary
                    if M in target_material_dict:
                        target_material_dict[M][1] += d_mole
                    else:
                        target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]

                    self._material_dict.pop(M)
                    self._layers_position_dict.pop(M)

                    # update solute_dict for both vessels
                    for Solute in copy.deepcopy(self._solute_dict):
                        if M in self._solute_dict[Solute]:
                            # calculate the amount to drain
                            d_mole = self._solute_dict[Solute][M]

                            # store the change of solute in d_solute
                            d_solute[Solute] += d_mole

                            # if all solutes are used up, eliminate the solute dictionary
                            if Solute not in target_solute_dict:
                                target_solute_dict[Solute] = {}

                            # update the target solute dictionary
                            if M in target_solute_dict[Solute]:
                                target_solute_dict[Solute][M] += d_mole
                            else:
                                target_solute_dict[Solute][M] = d_mole

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
                    if abs(self._solute_dict[Solute][Solvent] - 0.0) > 1e-6:
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
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
            # if M no longer in solute dict, which means all of this solute is drained
            else:
                d_mole = d_solute[M]

                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]

                # pop it from material dict since all drained out
                self._material_dict.pop(M)

        # deal with overflow
        v_max = target_vessel.get_max_volume()
        target_material_dict, target_solute_dict, temp_reward = util.check_overflow(
            material_dict=target_material_dict,
            solute_dict=target_solute_dict,
            v_max=v_max
        )

        reward += temp_reward

        print(
            '-------{}: {} (new_material_dict)-------'.format(
                self.label,
                self._material_dict
            )
        )
        print(
            '-------{}: {} (new_solute_dict)-------'.format(
                self.label,
                self._solute_dict
            )
        )
        print(
            '-------{}: {} (new_material_dict)-------'.format(
                target_vessel.label,
                target_material_dict
            )
        )
        print(
            '-------{}: {} (new_solute_dict)-------'.format(
                target_vessel.label,
                target_solute_dict
            )
        )

        # update target vessel's material amount
        event_1 = ['update material dict', target_material_dict]

        # update target vessel's solute dict
        event_2 = ['update solute dict', target_solute_dict]

        # create a event to reset material's position and variance in target vessel
        event_3 = ['fully mix']

        # push event to target vessel
        __ = target_vessel.push_event_to_queue(
            events=None,
            feedback=[event_1, event_2, event_3],
            dt=dt
        )

        return reward

    def _fully_mix(
            self,
            parameter,
            dt
    ):
        '''
        Method to completely mix a beaker
        '''

        # create new layer_position_dict
        new_layers_position_dict = {}

        # iterate through each material
        for M in self._material_dict:
            # do not fill in solute (solute does not form layer)
            if not self._material_dict[M][0]().is_solute():
                new_layers_position_dict[M] = 0.0

        # Add Air
        new_layers_position_dict[self.Air.get_name()] = 0.0

        # reset the variances
        self._layers_variance = 2.0

        # update self._layers_position_dict
        self._layers_position_dict = new_layers_position_dict

    def _update_material_dict(
            self,
            parameter,  # [new_material_dict]
            dt
    ):
        '''
        Method to update the material dictionary.
        '''

        new_material_dict = parameter[0]

        self._material_dict = util.organize_material_dict(new_material_dict)

    def _update_solute_dict(
            self,
            parameter,  # [new_solute_dict]
            dt
    ):
        '''
        '''

        new_solute_dict = parameter[0]

        # organize the target_solute_dict so it includes all solute and solvent
        self._solute_dict = util.organize_solute_dict(
            material_dict=self._material_dict,
            solute_dict=new_solute_dict
        )

    def _mix(
            self,
            parameter, # [mixing_parameter]
            dt
    ):
        '''
        Method to mix a vessel.
        '''

        # set the default reward to 0
        reward = 0

        # get mixing parameter from parameter if mixing, or get dt as mixing parameter for settling
        mixing_parameter = parameter[0] if parameter[0] else dt

        # collect data (according to separate.mix()):

        # convert the material dictionary to volumes in mL
        self_volume_dict, __ = util.convert_material_dict_to_volume(self._material_dict)

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
                solute_polarity.append(self._material_dict[Solute][0]().get_polarity())
        else:
            solute_amount.append([])
            solute_polarity.append(0.0)

        for M in self._layers_position_dict:  # if M forms a layer
            if M == 'Air':  # skip Air for now
                continue

            # if material exists in dictionary
            if M in self._material_dict.keys():
                # if the material is a solvent
                if self._material_dict[M][0]().is_solvent():
                    solvent_volume.append(self_volume_dict[M])
                    solvent_polarity.append(self._material_dict[M][0]().get_polarity())
                    solute_counter = 0

                    if self._solute_dict:
                        # fill in solute_amount
                        for Solute in self._solute_dict:
                            solute_amount[solute_counter].append(self._solute_dict[Solute][M])
                            solute_counter += 1
                    else:
                        solute_amount[0].append(0.0)

            # if material exists in dictionary
            if M in self._material_dict.keys():
                layers_position.append(self._layers_position_dict[M])
                layers_density.append(self._material_dict[M][0]().get_density())

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
                if self._material_dict[M][0]().is_solvent():
                    solute_counter = 0  # count position for solute in new_solute_amount
                    for Solute in self._solute_dict:
                        solute_amount = new_solute_amount[solute_counter][solvent_counter]
                        self._solute_dict[Solute][M] = solute_amount
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
        '''
        Modify self._layers.
        '''

        # collect data (according to separate.map())
        layers_amount = []
        layers_position = []
        layers_color = []
        layers_variance = self._layers_variance
        self_volume_dict, self_total_volume = util.convert_material_dict_to_volume(
            self._material_dict
        )

        for M in self._layers_position_dict:
            if M == 'Air':
                continue

            # if material exists in dictionary
            if M in self._material_dict.keys():
                layers_amount.append(self_volume_dict[M])
                layers_position.append((self._layers_position_dict[M]))
                layers_color.append(self._material_dict[M][0]().get_color())

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

    # functions to access private properties
    def get_material_amount(
            self,
            material_name=None
    ):
        '''
        :param material_name: a string or a list of strings
        :return:
        '''

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
        '''
        Method to obtain the volume of a material in the vessel.
        '''

        self_volume_dict, self_total_volume = util.convert_material_dict_to_volume(
            self._material_dict
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
                amount = 0.0

        return amount

    def get_material_position(
            self,
            material_name=None
    ):
        '''
        :param material_name: a string or a list of strings
        :return:
        '''

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
        '''
        Method to get the material variance property
        '''

        return self._layers_variance

    def get_layers(self):
        '''
        Method to get the vessel layers property
        '''

        return np.copy(self._layers)

    def get_material_color(self, material_name):
        '''
        Method to get the material color property
        '''

        return self._material_dict[material_name][0]().get_color()

    def get_material_dict(self):
        '''
        Method to get the material dictionary property
        '''

        return copy.deepcopy(self._material_dict)

    def get_solute_dict(self):
        '''
        Method to get the solute dictionary property
        '''

        return copy.deepcopy(self._solute_dict)

    def get_position_and_variance(
            self,
            dict_or_list='dict'
    ):
        '''
        Method to get the material position and variance property
        '''

        if dict_or_list == 'dict':
            position_list = copy.deepcopy(self._layers_position_dict)
        elif dict_or_list == 'list':
            position_list = []
            for L in self._layers_position_dict:
                position_list.append(self._layers_position_dict[L])

        return position_list, self._layers_variance

    def get_max_volume(self):
        '''
        Method to get the vessel's maximal volume property
        '''

        return self.v_max

    def get_min_volume(self):
        '''
        Method to get the vessel's minimal volume property
        '''

        return self.v_min

    def get_Tmin(self):
        '''
        Method to get the vessel's minimal temperature property
        '''

        return self.Tmin

    def get_Tmax(self):
        '''
        Method to get the vessel's maximal temperature property
        '''

        return self.Tmax

    def get_pmax(self):
        '''
        Method to get the vessel's maximal pressure property
        '''

        # set up a variable to contain the total pressure
        max_pressure = 0

        # calculate the total pressure in a vessel using the material dictionary
        for __, [__, material_amount] in self._material_dict.items():
            max_pressure += material_amount * R * self.Tmax / self.v_min

        # if the vessel contains no material use 1 atm as a baseline (in kPa)
        max_pressure = 101.325

        return max_pressure

    def get_defaultdt(self):
        '''
        Method to get the vessel's default time-step property
        '''

        return self.default_dt

    def get_temperature(self):
        '''
        Method to get the vessel's temperature property
        '''

        return self.temperature

    def get_pressure(self):
        '''
        Method to get the vessel's pressure property
        '''

        # set up a variable to contain the total pressure
        total_pressure = 0

        # calculate the total pressure in a vessel using the material dictionary
        for __, [__, material_amount] in self._material_dict.items():
            total_pressure += material_amount * R * self.temperature / self.volume

        return total_pressure

    def get_volume(self):
        '''
        Method to get the vessel's volume property
        '''

        return self.volume