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

        # switches for passive actions
        self.passive_settling = passive_settling
        self.passive_dissolving = passive_dissolving
        self.passive_heat_transfer = passive_heat_transfer

        self.air = material.Air()

        self._material_dict = {}  # material.name: material()
        self._material_amount_dict = {}  # material.name: amount

        self._solute_dict = {}  # solute.name: {solvent.name: percentage}

        # for vessel's pixel representation
        # self._pixel = np.zeros(100, dtype=np.float32) + self.air.color
        self._material_position_dict = {}  # material.name: position
        self._material_variance = 2.0

        # event queues
        self._event_dict = {'temperature change': self._update_temperature,
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
                            events,  # a list of event tuples(['event', parameters]), the last parameters is dt
                            ):
        if events is not None:
            self._event_queue.extend(events)
        self._update_materials()

    def _update_materials(self,
                          ):
        while self._event_queue:
            event = self._event_queue.pop(0)
            action = self._event_dict[event[0]]
            for i in self._material_dict.values():
                action(material=i, parameters=event[1:])
        merged = self.merge_event_queue(self._feedback_queue)
        while merged:
            feedback_event = merged.pop(0)
            action = self._event_dict[feedback_event[0]]
            for j in self._material_dict.values():
                action(material=j, parameters=feedback_event[1:])

    def merge_event_queue(self,  # need to be added
                          event_queue,
                          ):
        merged = []
        # merge events in the list, append merged and pop event_queue
        return merged

    # event functions
    def _update_temperature(self,  # this is an example,
                            material,
                            parameters,  # [target_temperature, dt]
                            ):
        self.temperature += parameters[0]
        events = material.update_temperature(target_temperature=parameters[0], dt=parameters[-1])
        self._feedback_queue.extend(events)

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
                    amount.append(self._material_amount_dict[i])
                except KeyError:
                    amount.append(0.0)
            return amount
        else:
            try:
                return self._material_amount_dict[material_name]
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