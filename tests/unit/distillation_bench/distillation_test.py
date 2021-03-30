import unittest
import sys
import os
sys.path.append('../../../')
sys.path.append('../../../chemistrylab/reactions')
import gym
from chemistrylab.reactions.available_reactions.chloro_wurtz import *
from chemistrylab.reactions.get_reactions import convert_to_class
import numpy as np

ENV_NAME = 'Distillation-v0'

'''
The following unittests should be ran with the dq parameter set in `distillation_bench_v1.py` set to 1000
'''

class DistillationTestCase(unittest.TestCase):
    '''
    Unit test class for distill_v0.py
    '''

    def test_init(self):
        env = gym.make(ENV_NAME)
        boil_vessel = env.boil_vessel
        target_material= env.target_material
        dQ=env.dQ
        n_actions=6

        self.assertEqual(boil_vessel, env.distillation.boil_vessel)
        self.assertEqual(target_material, env.distillation.target_material)
        self.assertEqual(dQ, env.distillation.dQ)
        self.assertEqual(n_actions, env.distillation.n_actions)

    def test_get_action_space(self):
        env = gym.make(ENV_NAME)
        action_space = gym.spaces.MultiDiscrete([6,10])
        self.assertEqual(action_space, env.distillation.get_action_space())

    def test_reset(self):
        env = gym.make(ENV_NAME)
        env.reset()
        num_of_vessels = 3
        materials={}
        solutes={}

        self.assertEqual(num_of_vessels, len(env.distillation.reset(env.boil_vessel)))
        self.assertIn('boil_vessel', env.distillation.reset(env.boil_vessel)[0].label)
        self.assertEqual(materials, env.distillation.reset(env.boil_vessel)[1]._material_dict)
        self.assertEqual(solutes, env.distillation.reset(env.boil_vessel)[2]._solute_dict)

    def test_temperature_increase(self):
        env=gym.make(ENV_NAME)
        env.reset()

        initial_temp = env.boil_vessel.temperature

        # setting action to increase temperature
        action = np.array([0,2000])
        env.step(action)

        desired_temp = initial_temp + env.boil_vessel.total_temp_change
        actual_temp = env.boil_vessel.temperature

        self.assertEqual(desired_temp,actual_temp)

    def test_pour_bv_b1(self):
        env=gym.make(ENV_NAME)
        env.reset()

        # gets material dict of bv before the pour
        bv_material_dict_before = env.boil_vessel.get_material_dict()

        # material list of bv before the pour
        # conversion to list is needed to compare with b1
        bv_material_list_before = []
        for key in bv_material_dict_before:
            bv_material_list_before.append(bv_material_dict_before[key][:2])

        # setting action to pour bv into b1
        action = np.array([1, 10])
        env.step(action)

        b1_material_list = []
        # converting into a list of material class and mol, no units
        for key in env.vessels[1]._material_dict:
            b1_material_list.append(env.vessels[1]._material_dict[key][:2])

        # env.boil_vessel._material_dict should be empty
        self.assertFalse(env.boil_vessel._material_dict)

        # beaker 1 should have the same material dict as boiling vessel before the pour
        self.assertEqual(bv_material_list_before, b1_material_list)

    # Reminder to create an issue for:
    # THE FOLLOWING SET OF ACTIONS BREAK DISTILLATION
    # Pour BV into B1
    # Pour B1 into B2
    # Pour B2 into BV
    # then try to heat it up, too many values to unpack (solvable by only checking the first 2 indices)

    # Documentation of plotting methods/others added in reaction base

    def test_pour_b1_b2(self):
        env = gym.make(ENV_NAME)
        env.reset()

        # boil off some materials into b1
        action = np.array([0, 500])
        env.step(action)

        # gets material dict of b1 before the pour
        b1_material_dict_before = env.vessels[1].get_material_dict()

        # material list of b1 before the pour
        # conversion to list is needed to compare with b2
        b1_material_list_before = []
        for key in b1_material_dict_before:
            b1_material_list_before.append(b1_material_dict_before[key][:2])

        # setting action to pour b1 into b2
        action = np.array([2, 10])
        env.step(action)

        b2_material_list = []
        # converting into a list of material class and mol, no units
        for key in env.vessels[2]._material_dict:
            b2_material_list.append(env.vessels[2]._material_dict[key][:2])

        # env.vessels[1]._material_dict should be empty
        self.assertFalse(env.vessels[1]._material_dict)

        # beaker 2 should have the same material dict as beaker 1 before the pour
        self.assertEqual(b1_material_list_before, b2_material_list)

    def test_pour_b1_bv(self):
        env = gym.make(ENV_NAME)
        env.reset()

        # boil off some materials into b1
        action = np.array([0, 500])
        env.step(action)

        # gets material dict of b1 before the pour
        # this will be added to bv_material_dict_before, in order to get the material_dict of bv after the pour
        b1_material_dict_before = env.vessels[1].get_material_dict()

        # gets material dict of bv before the pour
        bv_material_dict_before = env.boil_vessel.get_material_dict()

        print(b1_material_dict_before)
        print(bv_material_dict_before)

        # perform action
        # check that b1 is empty
        # add b1_material_dict_before to bv_material_dict before
        # check equal to current bv_material_dict

        # pour from b1 to bv
        action = np.array([3,10])
        env.step(action)

        # add b1_material_dict_before to bv_material_dict_before in order to get bv_material_dict after pour
        bv_material_dict_after = {**b1_material_dict_before, **bv_material_dict_before}

        # convert bv_dicts into lists
        bv_material_list_after = []
        for key in bv_material_dict_after:
            bv_material_list_after.append(bv_material_dict_after[key][:2])
        bv_material_list_after = sorted(bv_material_list_after, key=lambda l:l[1])

        bv_material_list_env = []
        for key in env.boil_vessel._material_dict:
            bv_material_list_env.append(env.boil_vessel._material_dict[key][:2])
        bv_material_list_env = sorted(bv_material_list_env, key=lambda l:l[1])

        # env.vessels[1]._material_dict should be empty
        self.assertFalse(env.vessels[1]._material_dict)

        # bv_material_dict_after should now have same material dict as boiling vessel
        self.assertEqual(bv_material_list_after, bv_material_list_env)

