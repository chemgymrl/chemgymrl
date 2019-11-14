import numpy as np
import math
from chemistrylab import *


class Vessel:
    def __init__(self,
                 label,
                 temperature=25,  # C
                 pressure=1,  # atm
                 v_max=1,
                 p_max=1.5,
                 open_vessel=True,  # True means no lid
                 max_feedback_iteration=1,
                 dt=0.05,
                 passive_settling=True,
                 passive_dissolving=True,
                 passive_heat_transfer=True,
                 ):
        self.label = label
        self.w2v = None
        self.temperature = temperature
        self.pressure = pressure
        self.v_max = v_max
        self.p_max = p_max
        self.open_vessel = open_vessel
        self.max_feedback_iteration = max_feedback_iteration
        self.dt = dt

        # switches for passive actions
        self.passive_settling = passive_settling
        self.passive_dissolving = passive_dissolving
        self.passive_heat_transfer = passive_heat_transfer

        self.air = material.Air()

        self._material_dict = {}  # material.name: [material(), amount]
        self._solute_dict = {}  # solute.name: {solvent.name: percentage}

        # for vessel's pixel representation
        # self._pixel = np.zeros(100, dtype=np.float32) + self.air.color
        self._material_position_dict = {}  # material.name: position
        self._material_variance = 2.0

        # event queues
        self._event_dict = {'temperature change': self._update_temperature,
                            'dissolve': self._dissolve,
                            }
        self._event_queue = []  # [['event', parameters], ['event', parameters] ... ]
        self._feedback_queue = []  # [same structure as event queue]

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

    def push_event_to_queue(self,
                            events,  # a list of event tuples(['event', parameters])
                            dt,
                            ):
        if events is not None:
            self._event_queue.extend(events)
        self._update_materials(dt)

    def _update_materials(self,
                          dt
                          ):
        while self._event_queue:
            event = self._event_queue.pop(0)
            action = self._event_dict[event[0]]
            action(parameters=event[1:], dt=dt)

        merged = self.merge_event_queue(self._feedback_queue, dt)
        while merged:
            feedback_event = merged.pop(0)
            action = self._event_dict[feedback_event[0]]
            action(parameters=feedback_event[1:], dt=dt)

    def merge_event_queue(self,  # need to be added
                          event_queue,
                          dt,
                          ):
        merged = []
        if self.passive_dissolving: merged.append(['dissolve'])
        # if any action that needs to be performed or checked each step, it can be added here

        # merge events in the list, append merged and pop the input event_queue
        return merged

    # event functions
    def _update_temperature(self,  # this is an example,
                            parameters,  # [target_temperature, dt]
                            dt,
                            ):
        feedback = []
        self.temperature += parameters[0]  # update vessel's property
        for i in self._material_dict.values():  # loop over all the materials
            feedback = i[0].update_temperature(target_temperature=parameters[0], dt=dt)
            self._feedback_queue.extend(feedback)

    def _dissolve(self,
                  dt):

    # a dissolve function that can figure out how materials are dissolved in others
    # solutes that are not fully dissolved will have a instance of that class stay in the self._material_dict
    # for the dissolved solute will be added to the self._solute_dict
    # example: NaCl dissolved in water and oil:
    # if not fully dissolved there will be NaCl in self._material_dict, if fully dissolved NaCl will be removed from it
    # and Na(charge=1), Cl(charge=-1) will be added to self._material_dict
    # and in the self._solute_dict, there should be {'Na': {'H2O': 0.8, 'C6H14': 0.2}, 'Cl':{'H2O': 0.8, 'C6H14': 0.2}}

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
                    position.append(self._material_position_dict[i])
                except KeyError:
                    position.append(0.0)
                    if i != 'Air':
                        print("error: requesting position for {} (non-existing material)".format(i))
            return position
        else:
            try:
                return self._material_position_dict[material_name]
            except KeyError:
                if material_name != 'Air':
                    print("error: requesting position for {} (non-existing material)".format(material_name))
                return 0.0

    def get_material_variance(self):
        return self._material_variance

    def get_pixel(self):
        return np.copy(self._pixel)
