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

ENV_NAME = 'WurtzReact-v1'

class DecompositionTestCase(unittest.TestCase):
    def test_init(self):
        env = gym.make(ENV_NAME)
        state = env.reset()
        self.assertEqual(True, bool(state.shape))

    def test_temperature_increase(self):
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 1
        action[1] = 1 / 2
        env.step(action)
        desired_temp = min(env.reaction.Ti + env.reaction.dT, env.reaction.Tmax)
        actual_temp = env.vessels.get_temperature()
        self.assertEqual(desired_temp, actual_temp)

    def test_volume_increase(self):
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 1 / 2
        action[1] = 1
        env.step(action)
        desired_volume = env.reaction.Vi + env.reaction.dV
        actual_volume = env.vessels.get_volume()
        self.assertAlmostEqual(desired_volume, actual_volume, places=6)

    def test_temperature_decrease(self):
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 0
        action[1] = 1 / 2
        env.step(action)
        desired_temp = max(env.reaction.Ti - env.reaction.dT, env.reaction.Tmin)
        actual_temp = env.vessels.get_temperature()
        self.assertEqual(desired_temp, actual_temp)

    def test_volume_decrease(self):
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 1 / 2
        action[1] = 0
        env.step(action)
        desired_volume = max(env.reaction.Vi - env.reaction.dV, env.reaction.Vmin)
        actual_volume = env.vessels.get_volume()
        self.assertAlmostEqual(desired_volume, actual_volume, places=6)

    def test_add_nacl(self):
        env = gym.make(ENV_NAME)
        state_before = env.reset()
        action = np.ones(env.action_space.shape)
        action[0] = 1 / 2
        action[1] = 1 / 2
        env.step(action)
        self.assertIn('Na', env.vessels.get_material_dict())
        self.assertIn('1-chlorohexane', env.vessels.get_material_dict())
        self.assertIn('2-chlorohexane', env.vessels.get_material_dict())
        self.assertIn('3-chlorohexane', env.vessels.get_material_dict())

    def test_full_render(self):
        env = gym.make(ENV_NAME)
        state_before = env.reset()
        env.render(mode="full")
        action = np.ones(env.action_space.shape)
        action[0] = 1 / 2
        action[1] = 1 / 2
        env.step(action)
        self.assertIn('Na', env.vessels.get_material_dict())
        self.assertIn('1-chlorohexane', env.vessels.get_material_dict())
        self.assertIn('2-chlorohexane', env.vessels.get_material_dict())
        self.assertIn('3-chlorohexane', env.vessels.get_material_dict())
        env.render(mode="full")

    def test_human_render(self):
        env = gym.make(ENV_NAME)
        state_before = env.reset()
        env.render()
        action = np.ones(env.action_space.shape)
        action[0] = 1 / 2
        action[1] = 1 / 2
        env.step(action)
        self.assertIn('Na', env.vessels.get_material_dict())
        self.assertIn('1-chlorohexane', env.vessels.get_material_dict())
        self.assertIn('2-chlorohexane', env.vessels.get_material_dict())
        self.assertIn('3-chlorohexane', env.vessels.get_material_dict())
        env.render()



if __name__ == '__main__':
    unittest.main()





