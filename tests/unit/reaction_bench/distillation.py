import unittest
import sys
sys.path.append('../')
sys.path.append('../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np


class DistillationTestCase(unittest.TestCase):
    def test_init(self):
        env = gym.make('DecompReactLesson-v0')
        state = env.reset()
        state_bool = bool(state)
        self.assertEqual(True, state_bool)

    def test_temperature_increase(self):
        env = gym.make('DecompReactLesson-v0')
        action = np.zeros(env.action_space.shape)
        action[0] = 1
        action[1] = 1/2
        env.step(action)
        desired_temp = env.Ti + env.dT
        actual_temp = env.vessel.get_temperature()
        self.assertEqual(desired_temp, actual_temp)

    def test_volume_increase(self):
        env = gym.make('DecompReactLesson-v0')
        action = np.zeros(env.action_space.shape)
        action[0] = 1/2
        action[1] = 1
        env.step(action)
        desired_volume = env.Vi + env.dV
        actual_volume = env.vessel.get_volume()
        self.assertEqual(desired_volume, actual_volume)

    def test_temperature_decrease(self):
        env = gym.make('DecompReactLesson-v0')
        action = np.zeros(env.action_space.shape)
        action[0] = 0
        action[1] = 1 / 2
        env.step(action)
        desired_temp = env.Ti - env.dT
        actual_temp = env.vessel.get_temperature()
        self.assertEqual(desired_temp, actual_temp)

    def test_volume_decrease(self):
        env = gym.make('DecompReactLesson-v0')
        action = np.zeros(env.action_space.shape)
        action[0] = 1 / 2
        action[1] = 0
        env.step(action)
        desired_volume = env.Vi - env.dV
        actual_volume = env.vessel.get_volume()
        self.assertEqual(desired_volume, actual_volume)

    def test_add_nacl(self):
        env = gym.make('DecompReactLesson-v0')
        state_before = env.reset()
        action = np.ones(env.action_space.shape)
        action[0] = 1 / 2
        action[1] = 1 / 2
        state_after, __ = env.step(action)
        self.assertNotEqual(state_after, state_before)


if __name__ == '__main__':
    unittest.main()
