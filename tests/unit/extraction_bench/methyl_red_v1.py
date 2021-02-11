import unittest
import sys
sys.path.append('../')
sys.path.append('../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np

ENV_NAME = 'MethylRed_Extract-v1'

class OilWaterTestCase(unittest.TestCase):
    def test_init(self):
        env = gym.make(ENV_NAME)
        state = env.reset()
        state_bool = bool(state)
        self.assertEqual(True, state_bool)

    def test_drain(self):
        env = gym.make(ENV_NAME)
        v0_initial = env.vessels[0].get_volume()
        v1_initial = env.vessels[1].get_volume()
        action = np.zeros(env.action_space.shape)
        action[0] = 0
        action[1] = 1
        vessels, ext_vessel, reward, done = env.step(action)
        self.assertLess(vessels[0].get_volume(), v0_initial)
        self.assertLess(v1_initial, vessels[1].get_volume())


    def test_mix_exv(self):
        env = gym.make(ENV_NAME)
        v0_initial_layers = env.vessels[0].get_layers()
        action = np.zeros(env.action_space.shape)
        action[0] = 1
        action[1] = 1
        vessels, ext_vessel, reward, done = env.step(action)
        self.assertLess(reward, 0)
        self.assertNotEqual(v0_initial_layers, vessels[0].get_layers())

    def test_mix_b1(self):
        env = gym.make(ENV_NAME)
        v1_initial_layers = env.vessels[1].get_layers()
        action = np.zeros(env.action_space.shape)
        action[0] = 2
        action[1] = 1
        vessels, ext_vessel, reward, done = env.step(action)
        self.assertLess(reward, 0)
        self.assertNotEqual(v1_initial_layers, vessels[1].get_layers())

    def test_mix_b2(self):
        env = gym.make(ENV_NAME)
        v2_initial_layers = env.vessels[2].get_layers()
        action = np.zeros(env.action_space.shape)
        action[0] = 3
        action[1] = 1
        vessels, ext_vessel, reward, done = env.step(action)
        self.assertLess(reward, 0)
        self.assertNotEqual(v2_initial_layers, vessels[2].get_layers())

    def test_add_b1_to_exv(self):
        env = gym.make(ENV_NAME)
        v0_initial = env.vessels[0].get_volume()
        v1_initial = env.vessels[1].get_volume()
        action = np.zeros(env.action_space.shape)
        action[0] = 4
        action[1] = 1
        vessels, ext_vessel, reward, done = env.step(action)
        self.assertLessEqual(v0_initial, vessels[0].get_volume())
        self.assertLessEqual(vessels[1].get_volume(), v1_initial)

    def test_add_b1_to_b2(self):
        env = gym.make(ENV_NAME)
        v0_initial = env.vessels[0].get_volume()
        v2_initial = env.vessels[2].get_volume()
        action = np.zeros(env.action_space.shape)
        action[0] = 5
        action[1] = 1
        vessels, ext_vessel, reward, done = env.step(action)
        self.assertLessEqual(v0_initial, vessels[0].get_volume())
        self.assertLessEqual(vessels[2].get_volume(), v2_initial)

    def test_add_exv_to_b2(self):
        env = gym.make(ENV_NAME)
        v0_initial = env.vessels[0].get_volume()
        v2_initial = env.vessels[2].get_volume()
        action = np.zeros(env.action_space.shape)
        action[0] = 6
        action[1] = 1
        vessels, ext_vessel, reward, done = env.step(action)
        self.assertLess(vessels[0].get_volume(), v0_initial)
        self.assertLess(v2_initial, vessels[2].get_volume())

    def test_add_oil_to_exv(self):
        env = gym.make(ENV_NAME)
        action = np.zeros(env.action_space.shape)
        action[0] = 7
        action[1] = 1
        vessels, ext_vessel, reward, done = env.step(action)
        self.assertIn('ethyl acetate', vessels[0].get_material_dict())

    def test_done(self):
        env = gym.make(ENV_NAME)
        action = np.zeros(env.action_space.shape)
        action[0] = 9
        action[1] = 1
        essels, ext_vessel, reward, done = env.step(action)
        self.assertEqual(done, True)


if __name__ == '__main__':
    unittest.main()
