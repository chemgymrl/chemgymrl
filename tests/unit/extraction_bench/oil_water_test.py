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
import unittest
import sys
sys.path.append('../../../')
sys.path.append('../../../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np
from chemistrylab.chem_algorithms.material import *

ENV_NAME = 'Oil_Water_Extract-v1'

"""
0: Valve (Speed multiplier, relative to max_valve_speed)
1: Mix ExV (mixing coefficient, *-1 when passed into mix function)
2: Mix B1 (mixing coefficient, *-1 when passed into mix function)
3: Mix B2 (mixing coefficient, *-1 when passed into mix function)
4: Add B1 to ExV (Volume multiplier, relative to max_vessel_volume)
5: Add B1 to B2 (Volume multiplier, relative to max_vessel_volume)
6: Add ExV to B2 （Volume multiplier, relative to max_vessel_volume)
7: Add Oil to ExV （Volume multiplier, relative to max_vessel_volume)
8: Wait (time multiplier)
9: Done (Value doesn't matter)
"""

class OilWaterTestCase(unittest.TestCase):
    def test_init(self):
        env = gym.make(ENV_NAME)
        state = env.reset()
        state_bool = bool(state)
        self.assertEqual(True, state_bool)

    def test_drain(self):
        env = gym.make(ENV_NAME)
        env.reset()
        v0_initial = env.vessels[0].get_current_volume()[-1]
        v1_initial = env.vessels[1].get_current_volume()[-1]
        action = np.zeros(env.action_space.shape)
        action[0] = 0
        action[1] = 1
        state, reward, done, _ = env.step(action)
        vessels = env.vessels
        self.assertLess(vessels[0].get_current_volume()[-1], v0_initial)
        self.assertLess(v1_initial, vessels[1].get_current_volume()[-1])

    def test_mix_exv(self):
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 7
        action[1] = 5
        state, reward, done, _ = env.step(action)

        # event_2 = ['fully mix']
        #
        # env.vessels[0].push_event_to_queue(events=None, feedback=[event_2], dt=0)

        v0_initial_layers = env.vessels[0].get_layers()

        action = np.zeros(env.action_space.shape)
        action[0] = 1
        action[1] = 5
        state, reward, done, _ = env.step(action)
        vessels = env.vessels
        equal = 0 in (v0_initial_layers == vessels[0].get_layers())
        self.assertTrue(equal)

    def test_mix_b1(self):
        env = gym.make(ENV_NAME)
        env.reset()

        material_dict = {'H2O': [H2O, 10, 'mol'], 'C6H14': [C6H14, 10, 'mol']}

        event_1 = ['update material dict', material_dict]
        event_2 = ['update_layer']

        env.vessels[1].push_event_to_queue(events=None, feedback=[event_1, event_2], dt=0)

        v1_initial_layers = env.vessels[1].get_layers()

        action = np.zeros(env.action_space.shape)
        action[0] = 2
        action[1] = 5
        state, reward, done, _ = env.step(action)
        vessels = env.vessels
        equal = 0 in (v1_initial_layers == vessels[1].get_layers())
        self.assertTrue(equal)

    def test_mix_b2(self):
        env = gym.make(ENV_NAME)
        env.reset()

        material_dict = {'H2O': [H2O, 10, 'mol'], 'C6H14': [C6H14, 10, 'mol']}
        event_1 = ['update material dict', material_dict]
        event_2 = ['update_layer']

        env.vessels[2].push_event_to_queue(events=None, feedback=[event_1, event_2], dt=0)

        v2_initial_layers = env.vessels[2].get_layers()
        action = np.zeros(env.action_space.shape)
        action[0] = 3
        action[1] = 5
        state, reward, done, _ = env.step(action)
        vessels = env.vessels
        equal = 0 in (v2_initial_layers == vessels[2].get_layers())
        self.assertTrue(equal)

    def test_mix_b2(self):
        env = gym.make(ENV_NAME)
        env.reset()

        material_dict = {'H2O': [H2O, 30], 'C6H14': [C6H14, 30]}

        event_1 = ['update material dict', material_dict]
        event_2 = ['update_layer']

        env.vessels[2].push_event_to_queue(events=None, feedback=[event_1, event_2], dt=0)


        v2_initial_layers = env.vessels[2].get_layers()
        action = np.zeros(env.action_space.shape)
        action[0] = 3
        action[1] = 5
        state, reward, done, _ = env.step(action)
        vessels = env.vessels
        equal = 0 in (v2_initial_layers == vessels[2].get_layers())
        self.assertTrue(equal)

    def test_add_b1_to_exv(self):
        env = gym.make(ENV_NAME)
        env.reset()
        v0_initial = env.vessels[0].get_current_volume()[-1]
        v1_initial = env.vessels[1].get_current_volume()[-1]
        action = np.zeros(env.action_space.shape)
        action[0] = 4
        action[1] = 5
        state, reward, done, _ = env.step(action)
        vessels = env.vessels
        self.assertLessEqual(v0_initial, vessels[0].get_current_volume()[-1])
        self.assertLessEqual(vessels[1].get_current_volume()[-1], v1_initial)

    def test_add_b1_to_b2(self):
        env = gym.make(ENV_NAME)
        env.reset()
        v0_initial = env.vessels[0].get_current_volume()[-1]
        v2_initial = env.vessels[2].get_current_volume()[-1]

        action = np.zeros(env.action_space.shape)
        action[0] = 5
        action[1] = 5
        state, reward, done, _ = env.step(action)
        vessels = env.vessels
        self.assertLessEqual(v0_initial, vessels[0].get_current_volume()[-1])
        self.assertLessEqual(vessels[2].get_current_volume()[-1], v2_initial)

    def test_add_exv_to_b2(self):
        env = gym.make(ENV_NAME)
        env.reset()

        action = np.zeros(env.action_space.shape)
        action[0] = 7
        action[1] = 5
        state, reward, done, _ = env.step(action)

        v0_initial = env.vessels[0].get_current_volume()[-1]
        v2_initial = env.vessels[2].get_current_volume()[-1]
        action = np.zeros(env.action_space.shape)
        action[0] = 6
        action[1] = 5
        state, reward, done, _ = env.step(action)
        vessels = env.vessels
        self.assertLess(vessels[0].get_current_volume()[-1], v0_initial)
        self.assertLess(v2_initial, vessels[2].get_current_volume()[-1])

    def test_add_oil_to_exv(self):
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 7
        action[1] = 5
        state, reward, done, _ = env.step(action)
        vessels = env.vessels
        self.assertIn('C6H14', vessels[0].get_material_dict())

    def test_full_render(self):
        env = gym.make(ENV_NAME)
        env.reset()
        env.render("full")
        action = np.zeros(env.action_space.shape)
        action[0] = 7
        action[1] = 5
        state, reward, done, _ = env.step(action)
        env.render("full")

    def test_human_render(self):
        env = gym.make(ENV_NAME)
        env.reset()
        env.render()
        action = np.zeros(env.action_space.shape)
        action[0] = 7
        action[1] = 5
        state, reward, done, _ = env.step(action)
        env.render()

    def test_done(self):
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 9
        action[1] = 1
        state, reward, done, _ = env.step(action)
        self.assertEqual(done, True)


if __name__ == '__main__':
    unittest.main()
