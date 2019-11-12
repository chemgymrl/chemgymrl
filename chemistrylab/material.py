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
        self.density = density # g/mL
        self.polarity = polarity
        self.temperature = temperature # K
        self.pressure = pressure # kPa
        self.phase = phase
        self.charge = charge
        self.molar_mass = molar_mass # g/mol
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
        super().__init__(name='Air',
                         density=1.225e-3,
                         temperature=297,
                         pressure=1,
                         phase='g',
                         molar_mass=28.963,
                         color=0.45
                         )


class H2O(Material):
    def __init__(self):
        super().__init__(name='H2O',
                         density=0.997,
                         polarity=2 * 1.24 * np.cos((109.5 / 2) * (np.pi / 180.0)),
                         temperature=298,
                         pressure=1,
                         phase='l',
                         molar_mass=18.015,
                         color=0.2,
                         charge=0.0
                         )


class C6H14(Material):
    def __init__(self):
        super().__init__(name='C6H14',
                         density=0.655,
                         polarity=0.0,
                         temperature=298,
                         pressure=1,
                         phase='l',
                         molar_mass=86.175,
                         color=0.65,
                         charge=0.0
                         )

class NaCl(Material):
    def __init__(self):
        super().__init__(name='NaCl',
                         density=2.165,
                         polarity=1.5,
                         temperature=298,
                         pressure=1,
                         phase='s',
                         molar_mass=58.443,
                         color=0.9,
                         charge=0.0
                         )

# Polarity is dependant on charge for atoms
class Na(Material):
    def __init__(self):
        super().__init__(name='Na',
                         density=0.968,
                         polarity=0.0,
                         temperature=298,
                         pressure=1,
                         phase='s',
                         molar_mass=22.990,
                         color=0.85,
                         charge=0.0
                         )

# Note: Cl is very unstable when not an aqueous ion
class Cl(Material):
    def __init__(self):
        super().__init__(name='Cl',
                         density=3.214e-3,
                         polarity=0.0,
                         temperature=298,
                         pressure=1,
                         phase='g',
                         molar_mass=35.453,
                         color=0.8,
                         charge=0.0
                         )

class Cl2(Material):
    def __init__(self):
        super().__init__(name='Cl2',
                         density=2.898e-3,
                         polarity=0.0,
                         temperature=298,
                         pressure=1,
                         phase='g',
                         molar_mass=70.906,
                         color=0.8,
                         charge=0.0
                         )

class LiF(Material):
    def __init__(self):
        super().__init__(name='LiF',
                         density=2.640,
                         polarity=1.5,
                         temperature=298,
                         pressure=1,
                         phase='s',
                         molar_mass=25.939,
                         color=0.9,
                         charge=0.0
                         )

class Li(Material):
    def __init__(self):
        super().__init__(name='Li',
                         density=0.534,
                         polarity=0.0,
                         temperature=298,
                         pressure=1,
                         phase='s',
                         molar_mass=6.941,
                         color=0.95,
                         charge=0.0
                         )

# Note: F is very unstable when not an aqueous ion
class F(Material):
    def __init__(self):
        super().__init__(name='F',
                         density=1.696e-3,
                         polarity=0.0,
                         temperature=298,
                         pressure=1,
                         phase='g',
                         molar_mass=18.998,
                         color=0.8,
                         charge=0.0
                         )

class F2(Material):
    def __init__(self):
        super().__init__(name='F2',
                         density=1.696e-3,
                         polarity=0.0,
                         temperature=298,
                         pressure=1,
                         phase='g',
                         molar_mass=37.997,
                         color=0.8,
                         charge=0.0
                         )

class T1(Material):
    def __init__(self):
        super().__init__(name='temp1',
                         density=0.655,
                         polarity=0.9,
                         temperature=298,
                         )


class T2(Material):
    def __init__(self):
        super().__init__(name='temp2',
                         density=0.655,
                         polarity=0.1,
                         temperature=298,
                         )
