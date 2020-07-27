import numpy as np
import math
from chemistrylab.chem_algorithms import material, util
from chemistrylab.extract_algorithms import separate
import copy


class Vessel:
    '''
    '''

    def __init__(self,
                 label,  # Name of the vessel
                 temperature=297,  # K
                 pressure=1.0,  # Kpa
                 volume=1000.0, # mL
                 materials={}, # moles of materials
                 solutes={}, # moles of solutes
                 v_max=1000.0,  # ml
                 v_min = 1000.0,
                 Tmax = 500,
                 Tmin = 250,
                 p_max=1.5,  # atm
                 default_dt=0.05,  # Default time for each step
                 n_pixels=100,  # Number of pixel to represent layers
                 open_vessel=True,  # True means no lid
                 # switches (used for defualt actions)
                 settling_switch=True,
                 layer_switch=True,
                 ):
        # initialize parameters
        self.label = label
        self.w2v = None
        self.temperature = temperature
        self.pressure = pressure
        self.volume = volume
        self.v_max = v_max
        self.v_min = v_min
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.p_max = p_max
        self.open_vessel = open_vessel
        self.default_dt = default_dt
        self.n_pixels = n_pixels

        # switches
        self.settling_switch = settling_switch
        self.layer_switch = layer_switch

        # create Air （not in material dict)
        self.Air = material.Air()

        # Material Dict and Solute Dict
        self._material_dict = materials  # material.name: [material(), amount]; amount is in mole
        self._solute_dict = solutes  # solute.name: [solvent(), amount]; amount is in mole

        # for vessel's pixel representation
        # layer's pixel representation, initially filled with Air's color
        self._layers = np.zeros(self.n_pixels, dtype=np.float32) + self.Air.get_color()
        # layers gaussian representation
        self._layers_position_dict = {self.Air.get_name(): 0.0}  # material.name: position
        self._layers_variance = 2.0

        # event dict holding all the event functions
        self._event_dict = {'temperature change': self._update_temperature,
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

    def push_event_to_queue(self,
                            events=None,  # a list of event tuples(['event', parameters])
                            feedback=None,  # a list of event tuples(['event', parameters])
                            dt=None,  # time for each step
                            check_switches_flag=True,  # whether check switches or not
                            ):
        reward = 0
        if dt is None:
            dt = self.default_dt
        if events is not None:  # if input event list is not None
            self._event_queue.extend(events)
        if feedback is not None:  # if input feedback list is not None
            self._feedback_queue.extend(feedback)
        reward = self._update_materials(dt, check_switches_flag)  # call self._update_materials() to execute events / feedback
        return reward

    def _update_materials(self,
                          dt,
                          check_switches_flag,
                          ):
        reward = 0
        # loop over event queue
        while self._event_queue:  # while it's not empty
            event = self._event_queue.pop(0)  # get the next event in event_queue
            print('-------{}: {} (event)-------'.format(self.label, event[0]))
            action = self._event_dict[event[0]]  # use the name of the event to get the function form event_dict
            reward += action(parameter=event[1:], dt=dt)  # call the event function

        # generate the merged queue from feedback queue and empty the feedback queue
        merged = self._merge_event_queue(self._feedback_queue, dt, check_switches_flag)
        self._feedback_queue = []

        # loop over merged queue
        while merged:  # while it's not empty
            feedback_event = merged.pop(0)  # get the next event in event_queue
            print('-------{}: {} (feedback_event)-------'.format(self.label, feedback_event[0]))
            action = self._event_dict[
                feedback_event[0]]  # use the name of the event to get the function form event_dict
            action(parameter=feedback_event[1:], dt=dt)  # call the event function
        return reward

    def _merge_event_queue(self,
                           event_queue,
                           dt,
                           check_switches_flag,
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
            # if any action that needs to be performed or checked each step, it can be added here

        return merged

    # event functions
    def _update_temperature(self,  # this is an example,
                            parameters,  # [target_temperature]
                            dt,
                            ):
        feedback = []
        self.temperature += parameters[0]  # update vessel's property
        for i in self._material_dict.values():  # loop over all the materials
            feedback = i[0].update_temperature(target_temperature=parameters[0], dt=dt)
            self._feedback_queue.extend(feedback)

    def _change_heat(self, parameter, dt):
        '''
        '''

        # extract the inputted parameters
        heat_change = parameter[0]
        out_beaker = parameter[1]

        print("Implement Heat Change of {:e} joules".format(heat_change))

        # create a list of increments of heat changes
        num_increments = 100
        heat_packets = [heat_change/num_increments for __ in range(num_increments)]

        # get the necessary vessel properties
        material_names = list(self._material_dict.keys())
        material_amounts = [amount for __, amount in self._material_dict.values()]
        material_objs = [material_obj for material_obj, __ in self._material_dict.values()]
        material_bps = [material_obj()._boiling_point for material_obj in material_objs]
        material_sp_heats = [material_obj()._specific_heat for material_obj in material_objs]

        # ensure all material boiling points are above the current vessel temperature
        if min(material_bps) < self.temperature:
            raise IOError("")

        # determine the material with the smallest boiling point and its value
        smallest_bp = min(material_bps)

        # modify the vessel temperature until a material's boiling point is reached;
        # then induce a phase change; once complete, continue adjusting the vessel temperature
        while self.temperature < smallest_bp:
            try:
                # obtain the change in heat
                dQ = heat_packets.pop(0)
            except IndexError:
                break

            # get parameters related to the material with the smallest boiling point
            smallest_bp_index = material_bps.index(smallest_bp)
            smallest_bp_material_name = material_names[smallest_bp_index]
            smallest_bp_material_amount = material_amounts[smallest_bp_index]
            smallest_bp_material_obj = material_objs[smallest_bp_index]
            smallest_bp_material_sp_heat = material_sp_heats[smallest_bp_index]

            # if the vessel temperature reaches a boiling point, boil off the material with that BP
            if all([
                self.temperature == smallest_bp,
                smallest_bp_material_amount > 0
            ]):
                # calculate the mass boiled off using the next packet of heat energy
                molar_mass = smallest_bp_material_obj()._molar_mass
                dM = dQ / (self.temperature * smallest_bp_material_sp_heat * molar_mass)

                # subtract the previous material amount by dM
                material_amounts[smallest_bp_index] -= dM
                print("Boiling Off {} mol of {}".format(dM, smallest_bp_material_name))

                # ensure the beaker has an entry for the boiled off material
                if not smallest_bp_material_name in out_beaker._material_dict.keys():
                    out_beaker._material_dict[smallest_bp_material_name] = [
                        smallest_bp_material_obj,
                        0.0
                    ]

                # add the boiled off material to the beaker
                out_beaker._material_dict[smallest_bp_material_name][1] += dM

            # if no material is left to boil off, update the smallest boiling point
            if all([
                self.temperature == smallest_bp,
                smallest_bp_material_amount == 0
            ]):
                # delete the boiled off material from all vessel-generated lists
                __ = material_names.pop(smallest_bp_index)
                __ = material_amounts.pop(smallest_bp_index)
                __ = material_objs.pop(smallest_bp_index)
                __ = material_sp_heats.pop(smallest_bp_index)

                # remove the previously smallest boiling point
                __ = material_bps.pop(smallest_bp_index)

                # update the smallest boiling point value
                smallest_bp = min(material_bps)

            # if no boiling point has been reached, modify the temperature of the vessel
            else:
                # calculate the total entropy of all the materials
                total_entropy = 0 # in J/K
                for i, material_amount in enumerate(material_amounts):
                    specific_heat = material_sp_heats[i] # in J/g*K
                    molar_amount = material_amount # in mol
                    molar_mass = material_objs[i]()._molar_mass # in g/mol

                    # calculate the entropy
                    total_entropy += specific_heat * molar_amount * molar_mass

                # calculate the change in temperature using dQ / (m * c) = dT
                dT = dQ / total_entropy
                self.temperature += dT

        print("Modified Vessel Temperature to {} Kelvin".format(self.temperature))

        # after completing the while loop, update the vessel's material dictionary
        new_material_dict = {}
        for i, name in enumerate(material_names):
            new_material_dict[name] = [material_objs[i], material_amounts[i]]
        self._material_dict = new_material_dict

        return 0

    def open_lid(self):
        if self.open_vessel:
            pass
        else:
            self.open_vessel = True
            self.pressure = 1
            # followed by several updates

    def close_lid(self):
        if not self.open_vessel:
            pass
        else:
            self.open_vessel = False
            # followed by several updates

    def _pour_by_volume(self,
                        parameter,  # [target_vessel, d_volume]
                        dt,
                        ):
        target_vessel = parameter[0]  # get the target vessel
        d_volume = parameter[1]  # must be greater than zero

        # collect data
        target_material_dict = target_vessel.get_material_dict()
        target_solute_dict = target_vessel.get_solute_dict()
        __, self_total_volume = util.convert_material_dict_to_volume(self._material_dict)

        reward = -1

        # if this vessel is empty or the d_volume is equal to zero, do nothing
        if math.isclose(self_total_volume, 0.0, rel_tol=1e-6) or math.isclose(d_volume, 0.0, rel_tol=1e-6):
            target_vessel.push_event_to_queue(events=None, feedback=None, dt=dt)
            return reward

        # if this vessel has enough volume of material to pour
        if self_total_volume - d_volume > 1e-6:
            # update material_dict for both vessels
            d_percentage = d_volume / self_total_volume  # calculate the percentage of amount of materials to be poured
            for M in copy.deepcopy(self._material_dict):
                d_mole = self._material_dict[M][
                             1] * d_percentage  # calculate the change of amount(mole) of each material
                self._material_dict[M][1] -= d_mole  # update material dict for this vessel
                # update material dict for the target vessel
                if M in target_material_dict:  # if it's in target vessel
                    target_material_dict[M][1] += d_mole
                else:  # if it's not in target vessel
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]),
                                               d_mole]  # create it in material dict

            # update solute_dict for both vessels
            for Solute in copy.deepcopy(self._solute_dict):
                if Solute not in target_solute_dict:  # if solute not in target vessel
                    target_solute_dict[Solute] = {}  # create it in solute dict
                for Solvent in self._solute_dict[Solute]:
                    d_mole = self._solute_dict[Solute][
                                 Solvent] * d_percentage  # calculate the change of amount in each solvent
                    self._solute_dict[Solute][Solvent] -= d_mole  # update self
                    if Solvent in target_solute_dict[Solute]:  # check if that solvent is in target material
                        target_solute_dict[Solute][Solvent] += d_mole
                    else:
                        target_solute_dict[Solute][Solvent] = d_mole
        else:  # if this vessel has less liquid than calculated amount then everything gets poured out
            # update material_dict for both vessels
            for M in copy.deepcopy(self._material_dict):
                d_mole = self._material_dict[M][1]  # pour all of this material
                if M in target_material_dict:  # check if this material is in target vessel
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
                self._material_dict.pop(M)  # remove it from this vessel
                if M in self._layers_position_dict:  # also remove it from self._layers_position_dict if it's in the dict
                    self._layers_position_dict.pop(M)
            # update solute_dict for both vessels
            for Solute in copy.deepcopy(self._solute_dict):
                if Solute not in target_solute_dict:  # if solute not in target vessel
                    target_solute_dict[Solute] = {}
                for Solvent in self._solute_dict[Solute]:
                    d_mole = self._solute_dict[Solute][Solvent]  # calculate the change of amount in each solvent
                    if Solvent in target_solute_dict[Solute]:  # check if that solvent is in target material
                        target_solute_dict[Solute][Solvent] += d_mole
                    else:
                        target_solute_dict[Solute][Solvent] = d_mole
                self._solute_dict.pop(Solute)  # pop the Solute

        # deal with overflow
        v_max = target_vessel.get_max_volume()  # get v_max from target vessel
        target_material_dict, target_solute_dict, temp_reward = util.check_overflow(material_dict=target_material_dict,
                                                                                    solute_dict=target_solute_dict,
                                                                                    v_max=v_max,
                                                                                    )
        reward += temp_reward
        # update target vessel's material amount
        event_1 = ['update material dict', target_material_dict]

        # update target vessel's solute dict
        event_2 = ['update solute dict', target_solute_dict]

        # create a event to reset material's position and variance in target vessel
        event_3 = ['fully mix']

        # push event to target vessel
        target_vessel.push_event_to_queue(events=None, feedback=[event_1, event_2, event_3], dt=dt)

        return reward

    def _drain_by_pixel(self,
                        parameter,  # [target_vessel, n_pixel]
                        dt,
                        ):
        target_vessel = parameter[0]
        n_pixel = parameter[1]

        # collect data
        target_material_dict = target_vessel.get_material_dict()
        target_solute_dict = target_vessel.get_solute_dict()
        __, self_total_volume = util.convert_material_dict_to_volume(self._material_dict)
        print('-------{}: {} (old_material_dict)-------'.format(self.label, self._material_dict))
        print('-------{}: {} (old_solute_dict)-------'.format(self.label, self._solute_dict))
        print('-------{}: {} (old_material_dict)-------'.format(target_vessel.label, target_material_dict))
        print('-------{}: {} (old_solute_dict)-------'.format(target_vessel.label, target_solute_dict))
        reward = -1

        # if this vessel is empty or the d_volume is equal to zero, do nothing
        if math.isclose(self_total_volume, 0.0, rel_tol=1e-6) or math.isclose(n_pixel, 0.0, rel_tol=1e-6):
            target_vessel.push_event_to_queue(events=None, feedback=None, dt=dt)
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
        for M in copy.deepcopy(self._material_dict):  # M is the solvent that's drained out
            if self._material_dict[M][0]().get_color() in color_to_pixel:  # if it's in color_to_pixel (means it get drained)
                # calculate the d_mole of this solvent
                d_volume = (color_to_pixel[self._material_dict[M][0]().get_color()] / self.n_pixels) * self.v_max

                # convert volume to amount (mole)
                density = self._material_dict[M][0]().get_density()
                molar_mass = self._material_dict[M][0]().get_molar_mass()
                d_mole = util.convert_volume_to_mole(volume=d_volume,
                                                     density=density,
                                                     molar_mass=molar_mass)

                if self._material_dict[M][1] - d_mole > 1e-6:  # if there are some of this solvent left
                    d_percentage = d_mole / self._material_dict[M][1]  # calculate the drained ratio
                    # update material dict for both vessels
                    self._material_dict[M][1] -= d_mole
                    if M in target_material_dict:  # check if it's in target vessel
                        target_material_dict[M][1] += d_mole
                    else:
                        target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
                    # update solute_dict for both vessels
                    for Solute in self._solute_dict:
                        if M in self._solute_dict[Solute]:
                            d_mole = self._solute_dict[Solute][M] * d_percentage  # calculate d_mole
                            self._solute_dict[Solute][M] -= d_mole
                            d_solute[Solute] += d_mole
                            if Solute not in target_solute_dict:
                                target_solute_dict[Solute] = {}
                            if M in target_solute_dict[Solute]:
                                target_solute_dict[Solute][M] += d_mole
                            else:
                                target_solute_dict[Solute][M] = d_mole
                else:  # if all drained out
                    # calculate the d_mole of this solvent
                    d_mole = self._material_dict[M][1]
                    if M in target_material_dict:  # check if it's in target vessel
                        target_material_dict[M][1] += d_mole
                    else:
                        target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
                    self._material_dict.pop(M)
                    self._layers_position_dict.pop(M)  # M must form layers, since it's in color_to_pixel
                    # update solute_dict for both vessels
                    for Solute in copy.deepcopy(self._solute_dict):
                        if M in self._solute_dict[Solute]:
                            d_mole = self._solute_dict[Solute][M]
                            d_solute[Solute] += d_mole  # store the change of solute in d_solute
                            if Solute not in target_solute_dict:  # check if solute in target vessel
                                target_solute_dict[Solute] = {}
                            if M in target_solute_dict[Solute]:
                                target_solute_dict[Solute][M] += d_mole
                            else:
                                target_solute_dict[Solute][M] = d_mole
                            self._solute_dict[Solute].pop(M)  # pop this solvent from solute

        # pop solute from solute_dict if solute_dict[solute] is empty or all the solvent in solute_dict[solute] is 0.0
        for Solute in copy.deepcopy(self._solute_dict):
            empty_flag = True
            if not self._solute_dict[Solute]:  # pop solute from solute dict if it's empty
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
            if M in self._solute_dict:  # if M still in solute dict, calculate the amount of it drained
                d_mole = d_solute[M]
                self._material_dict[M][1] -= d_mole
                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
            else:  # if M no longer in solute dict, which means all of this solute is drained
                d_mole = d_solute[M]
                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
                self._material_dict.pop(M)  # pop it from material dict since all drained out

        # deal with overflow
        v_max = target_vessel.get_max_volume()
        target_material_dict, target_solute_dict, temp_reward = util.check_overflow(material_dict=target_material_dict,
                                                                                    solute_dict=target_solute_dict,
                                                                                    v_max=v_max,
                                                                                    )
        reward += temp_reward
        print('-------{}: {} (new_material_dict)-------'.format(self.label, self._material_dict))
        print('-------{}: {} (new_solute_dict)-------'.format(self.label, self._solute_dict))
        print('-------{}: {} (new_material_dict)-------'.format(target_vessel.label, target_material_dict))
        print('-------{}: {} (new_solute_dict)-------'.format(target_vessel.label, target_solute_dict))
        # update target vessel's material amount
        event_1 = ['update material dict', target_material_dict]

        # update target vessel's solute dict
        event_2 = ['update solute dict', target_solute_dict]

        # create a event to reset material's position and variance in target vessel
        event_3 = ['fully mix']

        # push event to target vessel
        target_vessel.push_event_to_queue(events=None, feedback=[event_1, event_2, event_3], dt=dt)
        return reward

    def _fully_mix(self,
                   parameter,
                   dt,
                   ):
        # create new layer_position_dict
        new_layers_position_dict = {}
        for M in self._material_dict:
            if not self._material_dict[M][0]().is_solute():  # do not fill in solute (solute does not form layer)
                new_layers_position_dict[M] = 0.0
        new_layers_position_dict[self.Air.get_name()] = 0.0  # Add Air
        self._layers_variance = 2.0  # reset variances
        self._layers_position_dict = new_layers_position_dict  # update self._layers_position_dict

    def _update_material_dict(self,
                              parameter,  # [new_material_dict]
                              dt,
                              ):
        self._material_dict = util.organize_material_dict(parameter[0])  # organize material dict

    def _update_solute_dict(self,
                            parameter,  # [new_solute_dict]
                            dt,
                            ):
        # organize the target_solute_dict so it includes all solute and solvent
        self._solute_dict = util.organize_solute_dict(material_dict=self._material_dict,
                                                      solute_dict=parameter[0])

    def _mix(self,
             parameter,
             dt,
             ):
        reward = -1

        # get mixing parameter from parameter if mixing, or get dt as mixing parameter for settling
        if parameter[0] is None:
            mixing_parameter = dt
        else:
            mixing_parameter = parameter[0]

        # collect data (according to separate.mix()):
        self_volume_dict, __ = util.convert_material_dict_to_volume(self._material_dict)  # convert to ml
        solvent_volume = []
        layers_position = []
        layers_variance = self._layers_variance
        layers_density = []
        solute_polarity = []
        solvent_polarity = []
        solute_amount = []
        if self._solute_dict:
            for Solute in self._solute_dict:
                solute_amount.append([])  # append empty lists for each solute
                solute_polarity.append(self._material_dict[Solute][0]().get_polarity())  # collect polarity
        else:
            solute_amount.append([])
            solute_polarity.append(0.0)

        for M in self._layers_position_dict:  # if M forms a layer
            if M == 'Air':  # skip Air for now
                continue
            if self._material_dict[M][0]().is_solvent():  # if it's solvent
                solvent_volume.append(self_volume_dict[M])  # solvent amount
                solvent_polarity.append(self._material_dict[M][0]().get_polarity())  # solvent polairty
                solute_counter = 0
                if self._solute_dict:
                    for Solute in self._solute_dict:  # fill in solute_amount
                        solute_amount[solute_counter].append(self._solute_dict[Solute][M])
                        solute_counter += 1
                else:
                    solute_amount[0].append(0.0)

            layers_position.append(self._layers_position_dict[M])  # layers position
            layers_density.append(self._material_dict[M][0]().get_density())  # layers density

        # Add Air
        layers_position.append(self._layers_position_dict['Air'])
        layers_density.append(self.Air.get_density())

        '''
        print('solvent_volume (A): {}'.format(solvent_volume))
        print('layers_position (B): {}'.format(layers_position))
        print('variance: {}'.format(self._layers_variance))
        print('layers_density (D): {}'.format(layers_density))
        print('solute_polarity (Spol): {}'.format(solute_polarity))
        print('solvent_polarity (Lpol): {}'.format(solvent_polarity))
        print('solute_amount (S): {}'.format(solute_amount))
        print('mixing: {}'.format(mixing_parameter))
        '''
        new_layers_position, self._layers_variance, new_solute_amount, _ = separate.mix(A=np.array(solvent_volume),
                                                                                        B=np.array(layers_position),
                                                                                        C=layers_variance,
                                                                                        D=np.array(layers_density),
                                                                                        Spol=np.array(solute_polarity),
                                                                                        Lpol=np.array(solvent_polarity),
                                                                                        S=np.array(solute_amount),
                                                                                        mixing=mixing_parameter)
        '''
        print('new_layers_position (B): {}'.format(new_layers_position))
        print('new_solute_amount (S): {}'.format(new_solute_amount))
        print('new_variance: {}'.format(self._layers_variance))
        '''
        # use results returned from 'mix' to update self._layers_position_dict and self._solute_dict
        layers_counter = 0  # count position for layers in new_layers_position
        solvent_counter = 0  # count position for solvent in new_solute_amount for each solute
        for M in self._layers_position_dict:
            if M == 'Air':  # skip Air for now
                continue
            self._layers_position_dict[M] = new_layers_position[layers_counter]
            if self._solute_dict:
                layers_counter += 1
                if self._material_dict[M][0]().is_solvent:
                    solute_counter = 0  # count position for solute in new_solute_amount
                    for Solute in self._solute_dict:
                        self._solute_dict[Solute][M] = new_solute_amount[solute_counter][solvent_counter]
                        solute_counter += 1
                    solvent_counter += 1
        '''
        print('new_layers_position_dict : {}'.format(self._layers_position_dict))
        print('new_solute_dict (S): {}'.format(self._solute_dict))
        '''
        # deal with Air
        self._layers_position_dict['Air'] = new_layers_position[layers_counter]
        return reward

    def _update_layers(self,
                       parameter,
                       dt,
                       ):
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
    def get_material_amount(self,
                            material_name=None):
        """
        :param material_name: a string or a list of strings
        :return:
        """
        if type(material_name) is list:
            amount = []
            for i in material_name:
                try:
                    amount.append(self._material_dict[i][1])
                except KeyError:
                    amount.append(0.0)
            return amount
        else:
            try:
                return self._material_dict[material_name][1]
            except KeyError:
                return 0.0

    def get_material_volume(self,
                            material_name=None):
        self_volume_dict, self_total_volume = util.convert_material_dict_to_volume(self._material_dict)
        if type(material_name) is list:
            amount = []
            for i in material_name:
                try:
                    amount.append(self_volume_dict[i])
                except KeyError:
                    if material_name == 'Air':
                        return self.v_max - self_total_volume
                    amount.append(0.0)
            return amount
        else:
            try:
                return self_volume_dict[material_name]
            except KeyError:
                if material_name == 'Air':
                    return self.v_max - self_total_volume
                return 0.0

    def get_material_position(self,
                              material_name=None):
        """
        :param material_name: a string or a list of strings
        :return:
        """
        if type(material_name) is list:
            position = []
            for i in material_name:
                try:
                    position.append(self._layers_position_dict[i])
                except KeyError:
                    position.append(0.0)
                    if i != 'Air':
                        print("error: requesting position for {} (non-existing material)".format(i))
            return position
        else:
            try:
                return self._layers_position_dict[material_name]
            except KeyError:
                if material_name != 'Air':
                    print("error: requesting position for {} (non-existing material)".format(material_name))
                return 0.0

    def get_material_variance(self):
        return self._layers_variance

    def get_layers(self):
        return np.copy(self._layers)

    def get_material_color(self,
                           material):
        return self._material_dict[material][0]().get_color()

    def get_material_dict(self):
        return copy.deepcopy(self._material_dict)

    def get_solute_dict(self):
        return copy.deepcopy(self._solute_dict)

    def get_position_and_variance(self,
                                  dict_or_list='dict'):
        if dict_or_list == 'dict':
            return copy.deepcopy(self._layers_position_dict), self._layers_variance
        elif dict_or_list == 'list':
            position_list = []
            for L in self._layers_position_dict:
                position_list.append(self._layers_position_dict[L])
            return position_list, self._layers_variance

    def get_max_volume(self):
        return self.v_max

    def get_min_volume(self):
        return self.v_min

    def get_Tmin(self):
        return self.Tmin

    def get_Tmax(self):
        return self.Tmax

    def get_pmax(self):
        return self.p_max

    def get_defaultdt(self):
        return self.default_dt

    def get_temperature(self):
        return self.temperature
