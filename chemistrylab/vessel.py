import numpy as np
import math
from chemistrylab import material, util
from extract_algorithms import separate
import copy


class Vessel:
    def __init__(self,
                 label,
                 temperature=297,  # C
                 pressure=1,  # atm
                 v_max=1000,
                 p_max=1.5,
                 default_dt=0.05,
                 n_pixels=100,
                 open_vessel=True,  # True means no lid
                 settling_switch=True,  # switches
                 layer_switch=True,
                 ):
        self.label = label
        self.w2v = None
        self.temperature = temperature
        self.pressure = pressure
        self.v_max = v_max
        self.p_max = p_max
        self.open_vessel = open_vessel
        self.default_dt = default_dt
        self.n_pixels = n_pixels

        # switches
        self.settling_switch = settling_switch
        self.layer_switch = layer_switch

        self.Air = material.Air()

        self._material_dict = {}  # material.name: [material(), amount]
        self._solute_dict = {}  # solute.name: {solvent.name: amount}

        # for vessel's pixel representation
        self._layers = np.zeros(self.n_pixels, dtype=np.float32) + self.Air.get_color()
        self._layers_position_dict = {self.Air.get_name(): 0.0}  # material.name: position
        self._layers_variance = 2.0

        # event queues
        self._event_dict = {'temperature change': self._update_temperature,
                            'pour by volume': self._pour_by_volume,
                            'drain by pixel': self._drain_by_pixel,
                            'fully mix': self._fully_mix,
                            'update material dict': self._update_material_dict,
                            'update solute dict': self._update_solute_dict,
                            'mix': self._mix,
                            'update_layer': self._update_layers,
                            }
        self._event_queue = []  # [['event', parameters], ['event', parameters] ... ]
        self._feedback_queue = []  # [same structure as event queue]

    def push_event_to_queue(self,
                            events=None,  # a list of event tuples(['event', parameters])
                            feedback=None,
                            dt=0.05,
                            check_flags=True,
                            ):
        reward = 0
        if events is not None:
            self._event_queue.extend(events)
        if feedback is not None:
            self._feedback_queue.extend(feedback)
        reward = self._update_materials(dt, check_flags)
        return reward

    def _update_materials(self,
                          dt,
                          check_flags,
                          ):
        reward = 0
        print('-----------------{}----------------------'.format(self.label))
        while self._event_queue:
            event = self._event_queue.pop(0)
            print(event[0])
            action = self._event_dict[event[0]]
            reward += action(parameter=event[1:], dt=dt)
        merged = self._merge_event_queue(self._feedback_queue, dt, check_flags)
        self._feedback_queue = []
        while merged:
            feedback_event = merged.pop(0)
            print(feedback_event[0])
            action = self._event_dict[feedback_event[0]]
            action(parameter=feedback_event[1:], dt=dt)
        return reward

    def _merge_event_queue(self,  # need to be added
                           event_queue,
                           dt,
                           check_flags,
                           ):
        # TBD: settle, state

        merged = copy.deepcopy(event_queue)
        if check_flags:
            if self.settling_switch:
                merged.append(['mix', None])
            if self.layer_switch:
                merged.append(['update_layer'])
        # if any action that needs to be performed or checked each step, it can be added here

        # merge events in the list, append merged and pop the input event_queue
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
        target_vessel = parameter[0]
        d_volume = parameter[1]  # must be greater than zero
        pour_nothing = False  # a flag to check is poured nothing

        # collect data
        target_material_dict = target_vessel.get_material_dict()
        target_solute_dict = target_vessel.get_solute_dict()
        self_volume_dict, self_total_volume = util.convert_material_dict_to_volume(self._material_dict)

        reward = -1

        # if this vessel is empty
        if math.isclose(self_total_volume, 0.0, rel_tol=1e-5) or math.isclose(d_volume, 0.0, rel_tol=1e-5):
            print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
            target_vessel.push_event_to_queue(events=None, feedback=None, dt=dt)
            return reward

        # if this vessel has enough volume of material to pour
        if self_total_volume - d_volume > 1e-6:
            # update material_dict for both vessels
            d_percentage = d_volume / self_total_volume
            for M in copy.deepcopy(self._material_dict):
                d_mole = self._material_dict[M][1] * d_percentage
                self._material_dict[M][1] -= d_mole
                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
            # update solute_dict for both vessels
            for Solute in copy.deepcopy(self._solute_dict):
                if Solute not in target_solute_dict:
                    target_solute_dict[Solute] = {}
                for Solvent in self._solute_dict[Solute]:
                    d_mole = self._solute_dict[Solute][Solvent] * d_percentage
                    self._solute_dict[Solute][Solvent] -= d_mole
                    if Solvent in target_solute_dict[Solute]:
                        target_solute_dict[Solute][Solvent] += d_mole
                    else:
                        target_solute_dict[Solute][Solvent] = d_mole
        else:  # if this vessel has less liquid than calculated amount then everything gets poured out
            # update material_dict for both vessels
            for M in copy.deepcopy(self._material_dict):
                d_mole = self._material_dict[M][1]
                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
                self._material_dict.pop(M)
                if M in self._layers_position_dict:
                    self._layers_position_dict.pop(M)
            # update solute_dict for both vessels
            for Solute in copy.deepcopy(self._solute_dict):
                if Solute not in target_solute_dict:
                    target_solute_dict[Solute] = {}
                for Solvent in self._solute_dict[Solute]:
                    d_mole = self._solute_dict[Solute][Solvent]
                    self._solute_dict[Solute][Solvent] = 0.0
                    if Solvent in target_solute_dict[Solute]:
                        target_solute_dict[Solute][Solvent] += d_mole
                    else:
                        target_solute_dict[Solute][Solvent] = d_mole
                self._solute_dict.pop(Solute)

        # deal with overflow
        v_max = target_vessel.get_max_volume()
        target_material_dict, target_solute_dict = util.check_overflow(material_dict=target_material_dict,
                                                                       solute_dict=target_solute_dict,
                                                                       v_max=v_max,
                                                                       )
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
        drain_nothing = False  # a flag to check if drained nothing

        # collect data
        target_material_dict = target_vessel.get_material_dict()
        target_solute_dict = target_vessel.get_solute_dict()
        self_volume_dict, self_total_volume = util.convert_material_dict_to_volume(self._material_dict)

        reward = -1

        if math.isclose(self_total_volume, 0.0, rel_tol=1e-5) or math.isclose(n_pixel, 0.0, rel_tol=1e-5):
            target_vessel.push_event_to_queue(events=None, feedback=None, dt=dt)
            return reward

        # calculate pixels
        color_to_pixel = {}  # float as key may not be a good idea! maybe change this later
        (decimal, integer) = math.modf(n_pixel)
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
            if self._material_dict[M][0].get_color() in color_to_pixel:
                # calculate the d_mole of this solvent
                d_volume = (color_to_pixel[self._material_dict[M][0].get_color()] / self.n_pixels) * self.v_max
                density = self._material_dict[M][0].get_density()
                molar_mass = self._material_dict[M][0].get_molar_mass()
                d_mole = util.convert_volume_to_mole(volume=d_volume,
                                                     density=density,
                                                     molar_mass=molar_mass)
                if self._material_dict[M][1] - d_mole > 1e-6:  # if there are some of this solvent left
                    d_percentage = d_mole / self._material_dict[M][1]  # calculate the drained ratio
                    print('percentage: {}'.format(d_percentage))
                    # update material dict for both vessels
                    self._material_dict[M][1] -= d_mole
                    if M in target_material_dict:
                        target_material_dict[M][1] += d_mole
                    else:
                        target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
                    # update solute_dict for both vessels
                    for Solute in self._solute_dict:
                        if M in self._solute_dict[Solute]:
                            d_mole = self._solute_dict[Solute][M] * d_percentage
                            print('original: {}, {}: {}'.format(Solute, M,  self._solute_dict[Solute][M]))
                            print('delta: {}, {}: {}'.format(Solute, M, d_mole))
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
                    self._material_dict[M][1] = 0.0
                    if M in target_material_dict:
                        target_material_dict[M][1] += d_mole
                    else:
                        target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
                    self._material_dict.pop(M)
                    self._layers_position_dict.pop(M)  # M must form layers, since it's in color_to_pixel
                    # # update solute_dict for both vessels
                    for Solute in copy.deepcopy(self._solute_dict):
                        if M in self._solute_dict[Solute]:
                            d_mole = self._solute_dict[Solute][M]
                            self._solute_dict[Solute][M] = 0
                            d_solute[Solute] += d_mole
                            if Solute not in target_solute_dict:
                                target_solute_dict[Solute] = {}
                            if M in target_solute_dict[Solute]:
                                target_solute_dict[Solute][M] += d_mole
                            else:
                                target_solute_dict[Solute][M] = d_mole
                            self._solute_dict[Solute].pop(M)

        # pop solute from solute_dict if solute_dict[solute] is empty or all the solvent in solute_dict[solute] is 0
        for Solute in self._solute_dict:
            empty_flag = True
            if not self._solute_dict[Solute]:  # pop solute from solute dict if it's empty
                self._solute_dict.pop(Solute)
            else:
                for Solvent in self._solute_dict[Solute]:
                    if self._solute_dict[Solute][Solvent] - 0.0 > 1e-6:
                        empty_flag = False
            if empty_flag:
                self._solute_dict.pop(Solute)

        # use d_solute to update material dict for solute
        for M in d_solute:
            if M in self._solute_dict:
                print('{}: {}'.format(M, d_solute[M]))
                d_mole = d_solute[M]
                self._material_dict[M][1] -= d_mole
                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
            else:
                d_mole = self._material_dict[M][1]
                self._material_dict[M][1] = 0.0
                if M in target_material_dict:
                    target_material_dict[M][1] += d_mole
                else:
                    target_material_dict[M] = [copy.deepcopy(self._material_dict[M][0]), d_mole]
                self._material_dict.pop(M)

        # deal with overflow
        v_max = target_vessel.get_max_volume()
        target_material_dict, target_solute_dict = util.check_overflow(material_dict=target_material_dict,
                                                                       solute_dict=target_solute_dict,
                                                                       v_max=v_max,
                                                                       )

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
        new_layers_position_dict = {}
        for M in self._material_dict:
            if not self._material_dict[M][0].is_solute():  # not solute
                new_layers_position_dict[M] = 0.0
        new_layers_position_dict[self.Air.get_name()] = 0.0
        self._layers_variance = 2.0
        self._layers_position_dict = new_layers_position_dict

    def _update_material_dict(self,
                              parameter,  # [new_material_dict]
                              dt,
                              ):
        self._material_dict = util.organize_material_dict(parameter[0])

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
        # TBD: air? solid layer?
        # TBD: modify the mix function in separate

        # get mixing parameter from parameter if mixing, or get dt as mixing parameter for settling
        if parameter[0] is None:
            mixing_parameter = dt
        else:
            mixing_parameter = parameter[0]

        # collect data:
        self_volume_dict, self_total_volume = util.convert_material_dict_to_volume(self._material_dict)
        # print(self._material_dict)
        solvent_volume = []
        layers_position = []
        layers_variance = self._layers_variance
        layers_density = []
        solute_polarity = []
        solvent_polarity = []
        solute_amount = []
        if self._solute_dict:
            for Solute in self._solute_dict:  # append empty lists for each solute
                solute_amount.append([])
                solute_polarity.append(self._material_dict[Solute][0].get_polarity())

        else:
            solute_amount.append([])
            solute_polarity.append(0.0)

        for M in self._layers_position_dict:
            if M == 'Air':  # skip Air
                continue
            if self._material_dict[M][0].is_solvent():
                solvent_volume.append(self_volume_dict[M])  # solvent amount
                solvent_polarity.append(self._material_dict[M][0].get_polarity())
                solute_counter = 0
                if self._solute_dict:
                    for Solute in self._solute_dict:  # solute amount
                        solute_amount[solute_counter].append(self._solute_dict[Solute][M])
                        solute_counter += 1
                else:
                    solute_amount.append(0.0)

            layers_position.append(self._layers_position_dict[M])  # layers position
            layers_density.append(self._material_dict[M][0].get_density())  # layers density
        # Add Air
        layers_position.append(self._layers_position_dict['Air'])
        layers_density.append(self.Air.get_density())

        print(solvent_volume)
        print(layers_position)
        print(layers_density)
        print(solute_polarity)
        print(solvent_polarity)
        print(solute_amount)

        new_layers_position, self._layers_variance, new_solute_amount, _ = separate.mix(A=np.array(solvent_volume),
                                                                                        B=np.array(layers_position),
                                                                                        C=layers_variance,
                                                                                        D=np.array(layers_density),
                                                                                        Spol=np.array(solute_polarity),
                                                                                        Lpol=np.array(solvent_polarity),
                                                                                        S=np.array(solute_amount),
                                                                                        mixing=mixing_parameter)

        # use results returned from 'mix' to update self._layers_position_dict and self._solute_dict
        layers_counter = 0  # count position for layers in new_layers_position
        solvent_counter = 0  # count position for solvent in new_solute_amount for each solute
        for M in self._layers_position_dict:
            if M == 'Air':  # skip Air
                continue
            self._layers_position_dict[M] = new_layers_position[layers_counter]
            if self._solute_dict:
                layers_counter += 1  # count position for solute in new_solute_amount
                if self._material_dict[M][0].is_solvent:
                    solute_counter = 0
                    for Solute in self._solute_dict:
                        self._solute_dict[Solute][M] = new_solute_amount[solute_counter][solvent_counter]
                        solute_counter += 1
                    solvent_counter += 1
        # deal with Air
        self._layers_position_dict['Air'] = new_layers_position[layers_counter]

    def _update_layers(self,
                       parameter,
                       dt,
                       ):
        # collect data
        layers_amount = []
        layers_position = []
        layers_color = []
        layers_variance = self._layers_variance
        self_volume_dict, self_total_volume = util.convert_material_dict_to_volume(self._material_dict)
        print(self_total_volume)
        for M in self._layers_position_dict:
            if M == 'Air':
                continue
            layers_amount.append(self_volume_dict[M])
            layers_position.append((self._layers_position_dict[M]))
            layers_color.append(self._material_dict[M][0].get_color())

        # calculate air
        air_volume = self.v_max - self_total_volume
        if air_volume < 1e-6:
            air_volume = 0
        layers_amount.append(air_volume)
        layers_position.append(self._layers_position_dict['Air'])
        layers_color.append(self.Air.get_color())

        self._layers = separate.map_to_state(A=np.array(layers_amount),
                                             B=np.array(layers_position),
                                             C=layers_variance,
                                             colors=layers_color,
                                             x=separate.x,
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
        return self._material_dict[material][0].get_color()

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

