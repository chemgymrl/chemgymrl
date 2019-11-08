import numpy as np
import math


class Material:
    def __init__(self,
                 name=None,
                 density=None,
                 polarity=None,
                 temperature=None,
                 pressure=None,
                 phase=None,
                 charge=None,
                 molar_mass=None,
                 color=None,
                 ):
        self.name = name
        self.w2d = None
        self.density = density
        self.polarity = polarity
        self.temperature = temperature
        self.pressure = pressure
        self.phase = phase
        self.charge = charge
        self.molar_mass = molar_mass
        self.color = color

    def _update_properties(self,
                           # temperature,
                           # pressure,
                           # polarity,
                           # density,
                           # phase,
                           ):
        feedback = []
        # update the rest of properties
        # check if any feedback is generated, if so feedback.append(['event', parameter])
        return feedback

    def update_temperature(self, d_temperature):  # this is an example
        feedback = []
        self.temperature += d_temperature
        # check if a feedback is generated, if so feedback.append(['event', parameter])
        feedback.extend(self._update_properties())
        return feedback


class Air(Material):
    def __init__(self):
        super().__init__(name='C6H14',
                         color=0.45
                         )


class H2O(Material):
    def __init__(self):
        super().__init__(name='H2O',
                         density=0.997,
                         polarity=2 * 1.24 * np.cos((109.5 / 2) * (np.pi / 180.0)),
                         temperature=25,
                         pressure=1,
                         phase='l',
                         molar_mass=18.01528,
                         color=0.2,
                         charge=0.0
                         )


class C6H14(Material):
    def __init__(self):
        super().__init__(name='C6H14',
                         density=0.655,
                         polarity=0.0,
                         temperature=25,
                         pressure=1,
                         phase='l',
                         molar_mass=86.18,
                         color=0.65,
                         charge=0.0
                         )


class T1(Material):
    def __init__(self):
        super().__init__(name='temp1',
                         density=0.655,
                         polarity=0.9,
                         temperature=25,
                         )


class T2(Material):
    def __init__(self):
        super().__init__(name='temp2',
                         density=0.655,
                         polarity=0.1,
                         temperature=25,
                         )
