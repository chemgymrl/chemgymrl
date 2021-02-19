# import unittest
# import sys
# sys.path.append('../../../')
# sys.path.append('../../../chemistrylab/reactions')
# import gym
# import chemistrylab
# import numpy as np
#
# ENV_NAME = 'MethylRed_Extract-v2'
#
# """
# 0: Valve (Speed multiplier, relative to max_valve_speed)
# 1: Mix ExV (mixing coefficient, *-1 when passed into mix function)
# 2: Mix B1 (mixing coefficient, *-1 when passed into mix function)
# 3: Mix B2 (mixing coefficient, *-1 when passed into mix function)
# 4: Add B1 to ExV (Volume multiplier, relative to max_vessel_volume)
# 5: Add B1 to B2 (Volume multiplier, relative to max_vessel_volume)
# 6: Add ExV to B2 （Volume multiplier, relative to max_vessel_volume)
# 7: Add Ethyl Acetate to ExV （Volume multiplier, relative to max_vessel_volume)
# 8: Wait (time multiplier)
# 9: Done (Value doesn't matter)
# """
#
# class OilWaterTestCase(unittest.TestCase):
#     def test_init(self):
#         env = gym.make(ENV_NAME)
#         state = env.reset()
#         state_bool = bool(state)
#         self.assertEqual(True, state_bool)
#
#     def test_drain(self):
#         env = gym.make(ENV_NAME)
#         env.reset()
#         v0_initial = env.vessels[0].get_volume()
#         v1_initial = env.vessels[1].get_volume()
#         action = np.zeros(env.action_space.shape)
#         action[0] = 0
#         action[1] = 1
#         state, reward, done, _ = env.step(action)
#         vessels = env.vessels
#         print(vessels)
#         self.assertLess(vessels[0].get_volume(), v0_initial)
#         self.assertLess(v1_initial, vessels[1].get_volume())
#
#
#     def test_mix_exv(self):
#         env = gym.make(ENV_NAME)
#         env.reset()
#
#         action = np.zeros(env.action_space.shape)
#         action[0] = 7
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#
#         v0_initial_layers = env.vessels[0].get_layers()
#
#         action = np.zeros(env.action_space.shape)
#         action[0] = 1
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#         vessels = env.vessels
#         self.assertLess(reward, 0)
#         equal = 0 in (v0_initial_layers == vessels[0].get_layers())
#         self.assertTrue(equal)
#
#     def test_mix_b1(self):
#         env = gym.make(ENV_NAME)
#         env.reset()
#
#         action = np.zeros(env.action_space.shape)
#         action[0] = 7
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#
#         v1_initial_layers = env.vessels[1].get_layers()
#         action = np.zeros(env.action_space.shape)
#         action[0] = 2
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#         vessels = env.vessels
#         self.assertLess(reward, 0)
#         equal = 0 in (v1_initial_layers == vessels[1].get_layers())
#         self.assertTrue(equal)
#
#     def test_mix_b2(self):
#         env = gym.make(ENV_NAME)
#         env.reset()
#
#         action = np.zeros(env.action_space.shape)
#         action[0] = 7
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#
#         v2_initial_layers = env.vessels[2].get_layers()
#         action = np.zeros(env.action_space.shape)
#         action[0] = 3
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#         vessels = env.vessels
#         self.assertLess(reward, 0)
#         equal = 0 in (v2_initial_layers == vessels[2].get_layers())
#         self.assertTrue(equal)
#
#     def test_add_b1_to_exv(self):
#         env = gym.make(ENV_NAME)
#         env.reset()
#         v0_initial = env.vessels[0].get_volume()
#         v1_initial = env.vessels[1].get_volume()
#         action = np.zeros(env.action_space.shape)
#         action[0] = 4
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#         vessels = env.vessels
#         self.assertLessEqual(v0_initial, vessels[0].get_volume())
#         self.assertLessEqual(vessels[1].get_volume(), v1_initial)
#
#     def test_add_b1_to_b2(self):
#         env = gym.make(ENV_NAME)
#         env.reset()
#         v0_initial = env.vessels[0].get_volume()
#         v2_initial = env.vessels[2].get_volume()
#         action = np.zeros(env.action_space.shape)
#         action[0] = 5
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#         vessels = env.vessels
#         self.assertLessEqual(v0_initial, vessels[0].get_volume())
#         self.assertLessEqual(vessels[2].get_volume(), v2_initial)
#
#     def test_add_exv_to_b2(self):
#         env = gym.make(ENV_NAME)
#         env.reset()
#
#         action = np.zeros(env.action_space.shape)
#         action[0] = 7
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#
#         v0_initial = env.vessels[0].get_volume()
#         v2_initial = env.vessels[2].get_volume()
#         action = np.zeros(env.action_space.shape)
#         action[0] = 6
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#         vessels = env.vessels
#         self.assertLess(vessels[0].get_volume(), v0_initial)
#         self.assertLess(v2_initial, vessels[2].get_volume())
#
#     def test_add_oil_to_exv(self):
#         env = gym.make(ENV_NAME)
#         env.reset()
#         action = np.zeros(env.action_space.shape)
#         action[0] = 7
#         action[1] = 5
#         state, reward, done, _ = env.step(action)
#         vessels = env.vessels
#         self.assertIn('ethyl acetate', vessels[0].get_material_dict())
#
#     def test_done(self):
#         env = gym.make(ENV_NAME)
#         env.reset()
#         action = np.zeros(env.action_space.shape)
#         action[0] = 9
#         action[1] = 1
#         state, reward, done, _ = env.step(action)
#         self.assertEqual(done, True)
#
#
# if __name__ == '__main__':
#     unittest.main()
