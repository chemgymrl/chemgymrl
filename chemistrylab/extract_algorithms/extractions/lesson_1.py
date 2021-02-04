import numpy as np
import gym
from chemistrylab.extract_algorithms import separate
import math
import copy
from chemistrylab.chem_algorithms import material, util, vessel


class Extraction:
    def __init__(self,
                 extraction_vessel,
                 target_material,
                 n_empty_vessels=3,  # number of empty vessels
                 solution_volume=1000000000,  # the amount of oil available (unlimited)
                 dt=0.05,  # time for each step (time for separation)
                 max_vessel_volume=1000.0,  # max volume of empty vessels / g
                 n_vessel_pixels=100,  # number of pixels for each vessel
                 max_valve_speed=10,  # maximum draining speed (pixels/step)
                 n_actions=15,
                 solute=None,
                 extractor=None
                 ):
        # set self variable
        self.dt = dt
        self.n_actions = n_actions
        self.max_vessel_volume = max_vessel_volume
        self.solution_volume = solution_volume
        self.n_empty_vessels = n_empty_vessels
        self.n_total_vessels = n_empty_vessels + 1  # (empty + input)
        self.n_vessel_pixels = n_vessel_pixels
        self.max_valve_speed = max_valve_speed
        self.target_material = target_material
        self.target_material_init_amount = extraction_vessel.get_material_amount(target_material)

    # def get_observation_space(self):
    #     obs_low = np.zeros((self.n_total_vessels, self.n_vessel_pixels), dtype=np.float32)
    #     obs_high = 1.0 * np.ones((self.n_total_vessels, self.n_vessel_pixels), dtype=np.float32)
    #     observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
    #     return observation_space

    def get_action_space(self):
        """
        0: Valve (Speed multiplier, relative to max_valve_speed)
        1: Drain from ExV to B2
        1: Mix ExV (mixing coefficient, *-1 when passed into mix function)
        2: Mix B1 (mixing coefficient, *-1 when passed into mix function)
        3: Mix B2 (mixing coefficient, *-1 when passed into mix function)
        4: Mix B3 (mixing coefficient, *-1 when passed into mix function)
        4: Add B1 to ExV (Volume multiplier, relative to max_vessel_volume)
        5. Add B1 to B2
        5: Add B1 to B3 (Volume multiplier, relative to max_vessel_volume)
        6: Add ExV to B3 （Volume multiplier, relative to max_vessel_volume)
        7: Add solution 1 to to ExV （Volume multiplier, relative to max_vessel_volume)
        8: Add solution 2 to ExV （Volume multiplier, relative to max_vessel_volume)
        9: Do Nothing (waits dT)
        10: Done (Value doesn't matter)
        :return: action_space
        """
        # action_space = gym.spaces.Box(low=np.array([0, 0], dtype=np.float32),
        #                               high=np.array([self.n_actions, 1], dtype=np.float32),
        #                               dtype=np.float32)
        action_space = gym.spaces.MultiDiscrete([self.n_actions, 5])
        return action_space

    def reset(self, extraction_vessel):
        # generate empty vessels
        vessels = [copy.deepcopy(extraction_vessel)]
        for i in range(self.n_empty_vessels):
            temp_vessel = vessel.Vessel(label='beaker_{}'.format(i + 1),
                                        v_max=self.max_vessel_volume,
                                        default_dt=0.05,
                                        n_pixels=self.n_vessel_pixels
                                        )
            vessels.append(temp_vessel)

        # generate solution vessel2
        solution_1_vessel = vessel.Vessel(label='solution_1_vessel',
                                          v_max=self.solution_volume,
                                          n_pixels=self.n_vessel_pixels,
                                          settling_switch=False,
                                          layer_switch=False,
                                          )

        solution_2_vessel = vessel.Vessel(label='solution_2_vessel',
                                          v_max=self.solution_volume,
                                          n_pixels=self.n_vessel_pixels,
                                          settling_switch=False,
                                          layer_switch=False,
                                          )

        H2O = material.H2O
        Na = material.Na
        Cl = material.Cl
        Li = material.Li
        F = material.F

        solution_1_vessel_material_dict = {H2O().get_name(): [H2O, 30.0],
                                           Na().get_name(): [Na, 1.0],
                                           Cl().get_name(): [Cl, 1.0]
                                           }
        solution_1_solute_dict = {Na().get_name(): {H2O().get_name(): 1.0},
                                  Cl().get_name(): {H2O().get_name(): 1.0},
                                  }

        solution_2_vessel_material_dict = {H2O().get_name(): [H2O, 30.0],
                                           Li().get_name(): [Li, 1.0],
                                           F().get_name(): [F, 1.0]
                                           }
        solution_2_solute_dict = {Li().get_name(): {H2O().get_name(): 1.0},
                                  F().get_name(): {H2O().get_name(): 1.0},
                                  }

        solution_1_vessel_material_dict, solution_1_solute_dict, _ = util.check_overflow(
            material_dict=solution_1_vessel_material_dict,
            solute_dict=solution_1_solute_dict,
            v_max=solution_1_vessel.get_max_volume(),
            )

        solution_2_vessel_material_dict, solution_2_solute_dict, _ = util.check_overflow(
            material_dict=solution_2_vessel_material_dict,
            solute_dict=solution_2_solute_dict,
            v_max=solution_2_vessel.get_max_volume(),
        )

        event_s1_m = ['update material dict', solution_1_vessel_material_dict]
        event_s1_s = ['update solute dict', solution_1_solute_dict]

        event_s2_m = ['update material dict', solution_2_vessel_material_dict]
        event_s2_s = ['update solute dict', solution_2_solute_dict]

        solution_1_vessel.push_event_to_queue(feedback=[event_s1_m], dt=0)
        solution_1_vessel.push_event_to_queue(feedback=[event_s1_s], dt=0)

        solution_2_vessel.push_event_to_queue(feedback=[event_s2_m], dt=0)
        solution_2_vessel.push_event_to_queue(feedback=[event_s2_s], dt=0)

        return vessels, [solution_1_vessel, solution_2_vessel], util.generate_state(vessels, max_n_vessel=self.n_total_vessels)

    def perform_action(self, vessels, ext_vessel, action):
        """
        0: Valve (Speed multiplier, relative to max_valve_speed)
        1: Drain from ExV to B2
        2: Mix ExV (mixing coefficient, *-1 when passed into mix function)
        3: Mix B1 (mixing coefficient, *-1 when passed into mix function)
        4: Mix B2 (mixing coefficient, *-1 when passed into mix function)
        5: Mix B3 (mixing coefficient, *-1 when passed into mix function)
        6: Add B1 to ExV (Volume multiplier, relative to max_vessel_volume)
        7. Add B1 to B2
        8: Add B1 to B3 (Volume multiplier, relative to max_vessel_volume)
        9: Add ExV to B2 （Volume multiplier, relative to max_vessel_volume)
        10: Add ExV to B3 （Volume multiplier, relative to max_vessel_volume)
        11: Add solution 1 to to ExV （Volume multiplier, relative to max_vessel_volume)
        12: Add solution 2 to ExV （Volume multiplier, relative to max_vessel_volume)
        13: Do Nothing (waits dT)
        14: Done (Value doesn't matter)
        :param solution_1_vessel:
        :param solution_2_vessel:
        :param vessels: a list containing three vessels
        :param action: a tuple of two elements, first one represents action, second one is a multiplier
        :return: new vessels, oil vessel, reward, done
        """
        solution_1_vessel, solution_2_vessel = ext_vessel
        reward = -1  # default reward for each step
        done = False
        multiplier = (action[1]) / 4 if action[1] != 0 else 0
        # multiplier = action[1]
        if 0 == int(action[0]):  # 0: Valve (Speed multiplier)
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
            else:
                drained_pixels = multiplier * self.max_valve_speed
                event = ['drain by pixel', vessels[1], drained_pixels]
                reward = vessels[0].push_event_to_queue(events=[event], dt=self.dt)
            vessels[2].push_event_to_queue(dt=self.dt)

        if 1 == int(action[0]):  # Drain from ExV to B2
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
            else:
                drained_pixels = multiplier * self.max_valve_speed
                event = ['drain by pixel', vessels[2], drained_pixels]
                reward = vessels[0].push_event_to_queue(events=[event], dt=self.dt)
            vessels[2].push_event_to_queue(dt=self.dt)

        elif 2 == int(action[0]):  # 1: Mix ExV (TBD)
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                event = ['mix', -multiplier]
                reward = vessels[0].push_event_to_queue(events=[event], dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)

        elif 3 == int(action[0]):  # 2: Mix B1 (TBD)
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                event = ['mix', -multiplier]
                vessels[0].push_event_to_queue(dt=self.dt)
                reward = vessels[1].push_event_to_queue(events=[event], dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
        elif 4 == int(action[0]):  # 3: Mix B2 (TBD)
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                event = ['mix', -multiplier]
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                reward = vessels[2].push_event_to_queue(events=[event], dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)

        elif 5 == int(action[0]):  # 3: Mix B3 (TBD)
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                event = ['mix', -multiplier]
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                reward = vessels[3].push_event_to_queue(events=[event], dt=self.dt)

        elif 6 == int(action[0]):  # 4: Add B1 to ExV (Volume multiplier) [1, 0]
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[1].get_max_volume() * multiplier
                event = ['pour by volume', vessels[0], d_volume]
                reward = vessels[1].push_event_to_queue(events=[event], dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)

        elif 7 == int(action[0]):  # 5: Add B1 to B2 (Volume multiplier) [1, 2]
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[1].get_max_volume() * multiplier
                event = ['pour by volume', vessels[2], d_volume]
                reward = vessels[1].push_event_to_queue(events=[event], dt=self.dt)
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)

        elif 8 == int(action[0]):  # 5: Add B1 to B3 (Volume multiplier) [1, 2]
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[1].get_max_volume() * multiplier
                event = ['pour by volume', vessels[3], d_volume]
                reward = vessels[1].push_event_to_queue(events=[event], dt=self.dt)
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)

        elif 9 == int(action[0]):  # 6: Add ExV to B2 (Volume multiplier) [0, 2]
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[0].get_max_volume() * multiplier
                event = ['pour by volume', vessels[2], d_volume]
                reward = vessels[0].push_event_to_queue(events=[event], dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)

        elif 9 == int(action[0]):  # 6: Add ExV to B2 (Volume multiplier) [0, 2]
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[0].get_max_volume() * multiplier
                event = ['pour by volume', vessels[2], d_volume]
                reward = vessels[0].push_event_to_queue(events=[event], dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)

        elif 10 == int(action[0]):  # 6: Add ExV to B3 (Volume multiplier) [0, 2]
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[0].get_max_volume() * multiplier
                event = ['pour by volume', vessels[3], d_volume]
                reward = vessels[0].push_event_to_queue(events=[event], dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)

        elif 11 == int(action[0]):  # 7: Add solution 1 to ExV (Volume multiplier)
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[0].get_max_volume() * multiplier
                event = ['pour by volume', vessels[0], d_volume]
                reward = solution_1_vessel.push_event_to_queue(events=[event], dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)

        elif 12 == int(action[0]):  # 7: Add solution 2 to ExV (Volume multiplier)
            if multiplier == 0:
                vessels[0].push_event_to_queue(dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)
            else:
                d_volume = vessels[0].get_max_volume() * multiplier
                event = ['pour by volume', vessels[0], d_volume]
                reward = solution_2_vessel.push_event_to_queue(events=[event], dt=self.dt)
                vessels[1].push_event_to_queue(dt=self.dt)
                vessels[2].push_event_to_queue(dt=self.dt)
                vessels[3].push_event_to_queue(dt=self.dt)

        elif 13 == int(action[0]):  # do nothing
            wait_time = self.dt*(action[1] + 1)
            vessels[0].push_event_to_queue(dt=wait_time)
            vessels[1].push_event_to_queue(dt=wait_time)
            vessels[2].push_event_to_queue(dt=wait_time)
            vessels[3].push_event_to_queue(dt=wait_time)

        elif 14 == int(action[0]):  # 8: Done (Value doesn't matter)
            done = True
            reward = self.done_reward(vessels[2], vessels[3])

        return vessels, [solution_1_vessel, solution_2_vessel], reward, done

    def done_reward(self, beaker_2, beaker_3):  # beaker 2 is the beaker used to collect extracted material
        reward = 0
        if abs(beaker_2.get_material_amount(self.target_material[0]) - 0) < 1e-6:
            reward -= 100

        elif abs(beaker_3.get_material_amount(self.target_material[1]) - 0) < 1e-6:
            reward -= 100

        else:
            try:
                assert (abs(self.target_material_init_amount[0] - 0.0) > 1e-6)
                assert (abs(self.target_material_init_amount[1] - 0.0) > 1e-6)

                reward = (beaker_2.get_material_amount(self.target_material[0]) / self.target_material_init_amount[0]) * 100\
                         + (beaker_3.get_material_amount(self.target_material[1]) / self.target_material_init_amount[1]) * 100
                print("done_reward: {}, in_beaker_2: {}, initial: {}, in_beaker_3: {}, initial: {}"
                      .format(reward, beaker_2.get_material_amount(self.target_material), self.target_material_init_amount[0]),
                      beaker_3.get_material_amount(self.target_material[1]), self.target_material_init_amount[1])
            except AssertionError:
                reward = 0
                print("Oops! Division by zero. There's no target material in ExV(Extraction Vessel)")
        return reward