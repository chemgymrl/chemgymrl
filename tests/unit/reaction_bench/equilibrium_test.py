import unittest
import sys
sys.path.append('../../../')
sys.path.append('../../../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np

ENV_NAME = 'EquilReactLesson-v0'

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
        action[1] = 1/2
        env.step(action)
        desired_temp = env.Ti + env.dT
        actual_temp = env.vessels.get_temperature()
        self.assertEqual(desired_temp, actual_temp)

    def test_volume_increase(self):
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 1/2
        action[1] = 1
        env.step(action)
        desired_volume = env.Vi + env.dV
        actual_volume = env.vessels.get_volume()
        self.assertAlmostEqual(desired_volume, actual_volume, places=6)

    def test_temperature_decrease(self):
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 0
        action[1] = 1 / 2
        env.step(action)
        desired_temp = env.Ti - env.dT
        actual_temp = env.vessels.get_temperature()
        self.assertEqual(desired_temp, actual_temp)

    def test_volume_decrease(self):
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[0] = 1 / 2
        action[1] = 0
        env.step(action)
        desired_volume = env.Vi - env.dV
        actual_volume = env.vessels.get_volume()
        self.assertAlmostEqual(desired_volume, actual_volume, places=6)

    def test_add_nacl(self):
        env = gym.make(ENV_NAME)
        state_before = env.reset()
        # action = np.ones(env.action_space.shape)
        # action[0] = 1 / 2
        # action[1] = 1 / 2
        # env.step(action)
        self.assertIn('CuSO4', env.vessels.get_material_dict())
        self.assertIn('CuSO4*5H2O', env.vessels.get_material_dict())
        self.assertIn('H2O', env.vessels.get_material_dict())


if __name__ == '__main__':
    unittest.main()
