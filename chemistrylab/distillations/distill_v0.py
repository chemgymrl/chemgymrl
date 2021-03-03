 mshahen-2
'''
Module to perform actions to isolate a desired material generated in a Wurtz reaction.

:title: distill_v0.py

:author: Mitchell Shahen

:history: 2020-07-22

Available Actions for this Distillation Experiment:

0: Add/Remove Heat (Heat value multiplier, relative of maximal heat change)
1: Pour BV into B1 (Volume multiplier, relative to max_vessel_volume)
2: Pour B1 into B2 (Volume multiplier, relative to max_vessel_volume)
3: Pour B1 into BV (Volume multiplier, relative to max_vessel_volume)
4: Pour B2 into BV (Volume multiplier, relative to max_vessel_volume)
5: Done (Value doesn't matter)

Physical Representation:

A proper distillation experiment serves to extract one material from a vessel containing multiple
materials by utilizing the material's differing boiling points. Our distillation experiment
consists of three vessels: a distillation vessel which is placed on a source of heat and contains
all the material at the beginning of the experiment, the condensation vessel which is connected to
the distillation vessel via a tube and will contain the condensation of the material that is being
boiled off, and a storage vessel which is available such that the condensation vessel can be
emptied to make room for another evaporated material. The experiment begins by adding heat energy
to the distillation vessel which will serve to raise the temperature of the vessel and the
materials inside. The amount of energy required to raise the temperature of the distillation vessel
and the included materials is dependent on the specific heat capacities of the materials. Note: In
this virtual experiment, the vessel's overall heat capacity is calculated as the average heat
capacities of the materials in the vessel. As the temperature of the distillation vessel is
increased, we will eventually reach the lowest boiling point of the materials in the vessel. At
this point, any additional heat added will act to evaporate the material. The material, in it's
gaseous form, rises out of the distillation vessel and along the tube connecting the distillation
vessel to the condensation vessel, where the evaporated material condenses back into it's liquid
form. Note: It is assumed that the transfer of the gaseous material from the distillation vessel to
the condensation vessel happens instantaneously. Note: The amount of material evaporated is
dependent on the enthalpy of vapour of the material being evaporated. Once the entirety of the
material has been boiled off, the condensation vessel is drained into the storage vessel. Now that
the condensation beaker is empty, heat energy is once again added to the original beaker raising
its temperature until the next lowest boiling point is reached, thus repeating the above process.
The above process is repeated until the desired material has been completely evaporated out of the
distillation vessel into the condensation vessel, but before the condensation vessel has been
drained into the storage vessel. It is at this point that the desired material has been completely
isolated and we obtain a pure sample.
'''

# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position

import gym
import sys

sys.path.append("../../")
from chemistrylab.chem_algorithms import vessel, util

class Distillation:
    '''
    Class object for a Wurtz extraction experiment.

    Parameters
    ---------------
    `boil_vessel` : `vessel` (default=`None`)
        A vessel object containing state variables, materials, solutes, and spectral data.
    `target_material` : `str` (default=`None`)
        The name of the required output material designated as reward.
    `n_empty_vessels` : `int` (default=`2`)
        The number of empty vessels to add.
    `dQ` : `float` (default=`1.0`)
        The maximal change in heat energy.
    `dt` : `float` (default=`0.05`)
        The default time-step.
    `max_vessel_volume` : `float` (default=`1000.0`)
        The maximal volume of an added vessel.
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
            boil_vessel,
            target_material,
            n_empty_vessels=2,
            dQ=1.0, # maximal change in heat
            dt=0.05, # default time step
            max_vessel_volume=1.0, # maximal volume of empty vessels
            n_actions=6 # number of available actions
    ):
        '''
        Constructor class for the Distillation class
        '''

        # inputted vessels
        self.boil_vessel = boil_vessel

        # actions
        self.n_actions = n_actions

        # vessels
        self.n_empty_vessels = n_empty_vessels
        self.n_total_vessels = n_empty_vessels + 1
        self.max_vessel_volume = max_vessel_volume

        # target material
        self.target_material = target_material

        # thermodynamic variables
        self.dt = dt
        self.dQ = dQ

    def get_action_space(self):
        '''
        Method to describe the actions index and multipliers available.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `action_space` : `gym.spaces.MultiDiscrete`
            A MultiDiscrete object indicating the options for selecting actions and multipliers.

        Raises
        ---------------
        None
        '''

        # actions have two values:
        #   action[0] is the index of the action to perform
        #   action[1] is a multiplier for how much to do of that action
        action_space = gym.spaces.MultiDiscrete([self.n_actions, 10])

        return action_space

    def reset(self, boil_vessel):
        '''
        Method to reset the environment.

        Parameters
        ---------------
        `boil_vessel` : `vessel` (default=`None`)
            A vessel object containing state variables, materials, solutes, and spectral data.

        Returns
        ---------------
        `vessels` : `list`
            A list of all the vessel objects that contain materials and solutes.
        `state` : `np.array`
            An array containing state variables, material concentrations, and spectral data.

        Raises
        ---------------
        None
        '''

        # delete the solute dictionary from the boil vessel
        boil_vessel._solute_dict = {}

        # add the inputted boil vessel to the list of all vessels
        vessels = [boil_vessel]

        # create the empty beaker vessels, set variables to default values
        for i in range(self.n_empty_vessels):
            beaker = vessel.Vessel(
                label="beaker_{}".format(i),
                temperature=293.15, # room temp in Kelvin
                v_max=self.max_vessel_volume,
                default_dt=self.dt,
                materials={},
                solutes={}
            )
            vessels.append(beaker)

        return vessels

    def perform_action(self, vessels, action):
        '''
        Method to perform the action designated by `action` and update
        the state, vessels, external vessels, and generate reward.

        Parameters
        ---------------
        `vessels` : `list`
            A list of all the vessel objects that contain materials and solutes.
        `action` : `list`
            A list of two numbers indicating the index of an action to
            perform and a multiplier used when performin the action.

        Returns
        ---------------
        `vessels` : `list`
            A list of all the vessel objects that contain materials and solutes.
        `reward` : `float`
            The amount of the target material that has been generated by the most recent action.
        `done` : `bool`
            A boolean indicating if all of the required actions have now been completed.

        Raises
        ---------------
        None
        '''

        done = False

        # deconstruct the action
        do_action = int(action[0])
        multiplier = int(action[1])

        # if the multiplier is 0, only actions 0 (heat change) and 5 (done) can be performed
        if all([multiplier == 0, do_action not in [0, 5]]):
            for vessel_obj in vessels:
                __ = vessel_obj.push_event_to_queue(dt=self.dt)
                reward = 0
        else:
            # obtain the necessary vessels
            boil_vessel = vessels[0]
            beaker_1 = vessels[1]
            beaker_2 = vessels[2]

            # Add/Remove Heat (Heat multiplier)
            if do_action == 0:
                # calculate the amount of heat being added/removed
                multiplier = 2 * (multiplier/10 - 0.5)
                heat_change = multiplier * self.dQ

                # add the event to perform the heat change
                event = ['change_heat', heat_change, beaker_1]

                # push the event to the extraction vessel
                reward = boil_vessel.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to beaker 2
                __ = beaker_2.push_event_to_queue(dt=self.dt)

            # pour the Boil Vessel into Beaker 1
            if do_action == 1:
                # determine the volume to pour
                d_volume = boil_vessel.get_max_volume() * multiplier/10

                # push the event to the boil vessel
                event = ['pour by volume', beaker_1, d_volume]
                reward = boil_vessel.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the second beaker
                __ = beaker_2.push_event_to_queue(dt=self.dt)

            # Pour Beaker 1 into Beaker 2
            if do_action == 2:
                # determine the volume to pour
                d_volume = beaker_1.get_max_volume() * multiplier/10

                # push the event to the first beaker
                event = ['pour by volume', beaker_2, d_volume]
                reward = beaker_1.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the first beaker
                boil_vessel.push_event_to_queue(dt=self.dt)

            # Pour Beaker 1 into the boil vessel
            if do_action == 3:
                # determine the volume to pour
                d_volume = beaker_1.get_max_volume() * multiplier/10

                # push the event to the first beaker
                event = ['pour by volume', boil_vessel, d_volume]
                reward = beaker_1.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the second beaker
                beaker_2.push_event_to_queue(dt=self.dt)

            # Pour Beaker 2 into the boil vessel
            if do_action == 4:
                # determine the volume to pour
                d_volume = beaker_2.get_max_volume() * multiplier/10

                # push the event to the second beaker
                event = ['pour by volume', boil_vessel, d_volume]
                reward = beaker_2.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the first beaker
                beaker_1.push_event_to_queue(dt=self.dt)

            # Indicate that all no more actions are to be completed
            if do_action == 5:
                # pass the fulfilled `done` parameter
                done = True

                # look through each vessel's material dict looking for the target material
                reward = 0

        return vessels, reward, done
