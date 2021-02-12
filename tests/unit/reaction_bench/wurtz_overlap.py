import unittest
import sys
sys.path.append('../')
sys.path.append('../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np

REACTION_NAME = 'WurtzReact_overlap-v0'

class DecompositionTestCase(unittest.TestCase):
    def test_init(self):
        env = gym.make(REACTION_NAME)
        state = env.reset()
        state_bool = bool(state)
        self.assertEqual(True, state_bool)

    def test_temperature_increase(self):
        env = gym.make(REACTION_NAME)
        action = np.zeros(env.action_space.shape)
        action[0] = 1
        action[1] = 1/2
        env.step(action)
        desired_temp = env.Ti + env.dT
        actual_temp = env.vessel.get_temperature()
        self.assertEqual(desired_temp, actual_temp)

    def test_volume_increase(self):
        env = gym.make(REACTION_NAME)
        action = np.zeros(env.action_space.shape)
        action[0] = 1/2
        action[1] = 1
        env.step(action)
        desired_volume = env.Vi + env.dV
        actual_volume = env.vessel.get_volume()
        self.assertEqual(desired_volume, actual_volume)

    def test_temperature_decrease(self):
        env = gym.make(REACTION_NAME)
        action = np.zeros(env.action_space.shape)
        action[0] = 0
        action[1] = 1 / 2
        env.step(action)
        desired_temp = env.Ti - env.dT
        actual_temp = env.vessel.get_temperature()
        self.assertEqual(desired_temp, actual_temp)

    def test_volume_decrease(self):
        env = gym.make(REACTION_NAME)
        action = np.zeros(env.action_space.shape)
        action[0] = 1 / 2
        action[1] = 0
        env.step(action)
        desired_volume = env.Vi - env.dV
        actual_volume = env.vessel.get_volume()
        self.assertEqual(desired_volume, actual_volume)

    def test_add_all_materials(self):
        env = gym.make(REACTION_NAME)
        state_before = env.reset()
        action = np.ones(env.action_space.shape)
        action[0] = 1 / 2
        action[1] = 1 / 2
        env.step(action)
        self.assertIn('Na', env.vessel.get_material_dict())
        self.assertIn('1-chlorohexane', env.vessel.get_material_dict())


if __name__ == '__main__':
    unittest.main()
