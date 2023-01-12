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
"""
import numpy as np
import gym
import math
import copy
import sys

sys.path.append("../../../")
from chemistrylab.chem_algorithms import material, util, vessel
from chemistrylab.extract_algorithms import separate


class Extraction:
    '''
    '''

    def __init__(
        self,
        extraction_vessel,
        target_material,
        n_empty_vessels=2,  # number of empty vessels
        ethyl_volume=1000000000,  # the amount of oil available (unlimited)
        dt=0.05,  # time for each step (time for separation)
        max_vessel_volume=1.0,  # max volume of empty vessels / g
        n_vessel_pixels=100,  # number of pixels for each vessel
        max_valve_speed=10,  # maximum draining speed (pixels/step)
        n_actions=10,
        solvents=[""],
        extractor=None
    ):
        '''
        '''

        # set self variable
        self.dt = dt
        self.n_actions = n_actions
        self.max_vessel_volume = max_vessel_volume
        self.ethyl_volume = ethyl_volume
        self.n_empty_vessels = n_empty_vessels
        self.n_total_vessels = n_empty_vessels + 1  # (empty + input)
        self.n_vessel_pixels = n_vessel_pixels
        self.max_valve_speed = max_valve_speed
        self.target_material = target_material
        self.target_material_init_amount = extraction_vessel.get_material_amount(target_material)

    def get_observation_space(self,targets):
        #added targets as parameter (it increases the size of the obs space) -KS
        obs_low = np.zeros((self.n_total_vessels, self.n_vessel_pixels+len(targets)), dtype=np.float32)
        obs_high = 1.0 * np.ones((self.n_total_vessels, self.n_vessel_pixels+len(targets)), dtype=np.float32)
        observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        return observation_space

    def get_action_space(self):
        '''
        0: Valve (Speed multiplier, relative to max_valve_speed)
        1: Mix ExV (mixing coefficient, *-1 when passed into mix function)
        2: Mix B1 (mixing coefficient, *-1 when passed into mix function) # maybe never done
        3: Mix B2 (mixing coefficient, *-1 when passed into mix function) # maybe never done
        4: Add B1 to ExV (Volume multiplier, relative to max_vessel_volume)
        5: Add B1 to B2 (Volume multiplier, relative to max_vessel_volume)
        6: Add ExV to B2 （Volume multiplier, relative to max_vessel_volume)
        7: Add Oil to ExV （Volume multiplier, relative to max_vessel_volume)
        8: Wait (time multiplier)
        9: Done (Value doesn't matter)
        :return: action_space
        '''

        # action_space = gym.spaces.Box(low=np.array([0, 0], dtype=np.float32),
        #                               high=np.array([self.n_actions, 1], dtype=np.float32),
        #                               dtype=np.float32)

        action_space = gym.spaces.MultiDiscrete([self.n_actions, 5])

        return action_space

    def reset(self, extraction_vessel,targets):
        #added targets as parameter (it's added to the observation) -KS
        '''
        generate empty vessels
        '''

        vessels = [copy.deepcopy(extraction_vessel)]
        for i in range(self.n_empty_vessels):
            temp_vessel = vessel.Vessel(
                label='beaker_{}'.format(i + 1),
                v_max=self.max_vessel_volume,
                default_dt=0.05,
                n_pixels=self.n_vessel_pixels
            )
            vessels.append(temp_vessel)

        # generate oil vessel
        ethyl_vessel = vessel.Vessel(
            label='ethyl_vessel',
            v_max=self.ethyl_volume,
            n_pixels=self.n_vessel_pixels,
            settling_switch=False,
            layer_switch=False,
        )

        ethyl = material.EthylAcetate()

        ethyl_vessel_material_dict = {ethyl.get_name(): [ethyl, self.ethyl_volume]}

        ethyl_vessel_material_dict, _, _ = util.check_overflow(
            material_dict=ethyl_vessel_material_dict,
            solute_dict={},
            v_max=ethyl_vessel.get_max_volume()
        )
        event = ['update material dict', ethyl_vessel_material_dict]
        ethyl_vessel.push_event_to_queue(feedback=[event], dt=0)

        #state = util.generate_layers_obs(vessels, max_n_vessel=self.n_total_vessels, n_vessel_pixels=self.n_vessel_pixels)
        
        #state is now modified to include information about the target
        state = np.zeros((self.n_total_vessels, self.n_vessel_pixels + len(targets)), dtype=np.float32)

        # generate the part of the state indep of your trarget
        state[:, :self.n_vessel_pixels] = util.generate_layers_obs(
            vessel_list=vessels,
            max_n_vessel=self.n_total_vessels,
            n_vessel_pixels=self.n_vessel_pixels
        )
        #add in info about your target
        targ_ind = targets.index(self.target_material)
        state[:, self.n_vessel_pixels + targ_ind] += 1
        

        return vessels, ethyl_vessel, state

    def perform_action(self, vessels, ext_vessel, action):
        '''
        0: Valve (Speed multiplier, relative to max_valve_speed)
        1: Mix ExV (mixing coefficient, *-1 when passed into mix function)
        2: Mix B1 (mixing coefficient, *-1 when passed into mix function)
        3: Mix B2 (mixing coefficient, *-1 when passed into mix function)
        4: Add B1 to ExV (Volume multiplier, relative to max_vessel_volume)
        5: Add B1 to B2 (Volume multiplier, relative to max_vessel_volume)
        6: Add ExV to B2 （Volume multiplier, relative to max_vessel_volume)
        7: Add Ethyl Acetate to ExV （Volume multiplier, relative to max_vessel_volume)
        8: Wait (time multiplier)
        9: Done (Value doesn't matter)
        :param oil_vessel:
        :param vessels: a list containing three vessels
        :param action: a tuple of two elements, first one represents action, second one is a multiplier
        :return: new vessels, oil vessel, reward, done
        '''

        reward = 0 # default reward for each step
        done = False
        multiplier = (action[1]) / 4 if action[1] != 0 else 0
        # multiplier = action[1]

        # 0: Valve (Speed multiplier)
        if 0 == int(action[0]):
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
            else:
                drained_pixels = multiplier * self.max_valve_speed
                event = ['drain by pixel', vessels[1], drained_pixels]
                reward = vessels[0].push_event_to_queue(events=[event], dt=self.dt)
            vessels[2].push_event_to_queue(dt=self.dt)
        # 1: Mix ExV
        elif 1 == int(action[0]):
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
            else:
                event = ['mix', -multiplier]
                reward = vessels[0].push_event_to_queue(events=[event], dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
        # 2: Mix B1
        elif 2 == int(action[0]):
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
            else:
                event = ['mix', -multiplier]
                vessels[0].push_event_to_queue(dt=self.dt)
                reward = vessels[1].push_event_to_queue(events=[event], dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
        # 3: Mix B2
        elif 3 == int(action[0]):
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
            else:
                event = ['mix', -multiplier]
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                reward = vessels[2].push_event_to_queue(events=[event], dt=self.dt)
        # 4: Add B1 to ExV (Volume Multiplier) [1, 0]
        elif 4 == int(action[0]):
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[1].get_max_volume() * multiplier
                event = ['pour by volume', vessels[0], d_volume]
                reward = vessels[1].push_event_to_queue(events=[event], dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
        # 5: Add B1 to B2 (Volume multiplier) [1, 2]
        elif 5 == int(action[0]):
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[1].get_max_volume() * multiplier
                event = ['pour by volume', vessels[2], d_volume]
                reward = vessels[1].push_event_to_queue(events=[event], dt=self.dt)
                vessels[0].push_event_to_queue(dt=self.dt)
        # 6: Add ExV to B2 (Volume multiplier) [0, 2]
        elif 6 == int(action[0]):
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[0].get_max_volume() * multiplier
                event = ['pour by volume', vessels[2], d_volume]
                reward = vessels[0].push_event_to_queue(events=[event], dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
        # 7: Add ethyl acetate to ExV (Volume multiplier)
        elif 7 == int(action[0]):
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
            else:
                # add 50ml*multiplier
                d_volume = 0.050 * multiplier
                event = ['pour by volume', vessels[0], d_volume]
                reward = ext_vessel.push_event_to_queue(events=[event], dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
        elif 8 == int(action[0]):  # wait
            wait_time = self.dt*(action[1] + 1)
            vessels[0].push_event_to_queue(dt=wait_time)
            vessels[1].push_event_to_queue(dt=wait_time)
            vessels[2].push_event_to_queue(dt=wait_time)
        # 8: Done
        elif 9 == int(action[0]):
            done = True

        return vessels, ext_vessel, reward, done

    def done_reward(self, beaker_2):
        '''
        beaker 2 is the beaker used to collect extracted material
        '''

        if abs(beaker_2.get_material_amount(self.target_material) - 0) < 1e-6:
            reward = -100
        else:
            try:
                assert (abs(self.target_material_init_amount - 0.0) > 1e-6)
                reward = (beaker_2.get_material_amount(self.target_material) / self.target_material_init_amount) * 100
                print("done_reward: {}, in_beaker_2: {}, initial: {}".format(reward, beaker_2.get_material_amount(
                    self.target_material), self.target_material_init_amount))
            except AssertionError:
                reward = 0
                print("Oops! Division by zero. There's no target material in ExV(Extraction Vessel)")
        return reward
