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

        # REMINDER TO MAKE AN ISSUE THAT THERE'S NO ACTUAL WAY TO DECREASE HEAT OF BOILING VESSEL
        # ALL ACTIONS ARE DONE ONLY WHEN HEAT AVAILABLE > 0
        # CANNOT MAKE UNIT TEST FOR TEMP_DECREASE

    def test_pour_bv_b1(self):
        env=gym.make(ENV_NAME)
        env.reset()

        multiplier = 10
        change_in_volume = env.boil_vessel.get_max_volume() * multiplier / 10

        # setting action to increase temperature
        action = np.array([1, multiplier])
        env.step(action)