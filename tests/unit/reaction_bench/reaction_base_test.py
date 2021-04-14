import unittest
import sys
import os
sys.path.append('../../../')
sys.path.append('../../../chemistrylab/reactions')
import gym
from chemistrylab.reactions.available_reactions.chloro_wurtz import *
from chemistrylab.reactions.get_reactions import convert_to_class
import numpy as np
from gym import envs
from chemistrylab.reaction_bench.reaction_bench_v1 import ReactionBenchEnv_0

ENV_NAME = 'WurtzReact-v1'

class ReactionBaseTestCase(unittest.TestCase):
    '''
    Unit test class for reaction base using the chloro-wurtz.py reaction file
    '''

    def test_init_name(self):

        # testing for __init__ to see if name is set correctly
        env = gym.make(ENV_NAME)
        name = 'base_reaction'
        reaction_base_name = env.reaction.name
        self.assertEqual(name, reaction_base_name)

    def test_init_params(self):

        # testing to see if parameters for generating spectra are defined correctly
        env = gym.make(ENV_NAME)

        # converting materials to class object representation
        material_classes = convert_to_class(materials=REACTANTS+PRODUCTS)

        # define parameters for generating spectra
        params = []
        for material in material_classes:
            params.append(material().get_spectra_no_overlap())

        self.assertEqual(params, env.reaction.params)

    def test_find_reaction_file(self):

        # testing _find_reaction_file() method
        env = gym.make(ENV_NAME)
        # path = "../../chemistrylab/reactions/available_reactions"
        # # os.chdir(path)
        # file_path = path + '/chloro_wurtz.py'
        reaction_base_file_path = env.reaction._find_reaction_file(reaction_file="chloro_wurtz")

        # checks if file paths are the same
        # self.assertEqual(True, bool(os.path.samefile(file_path, reaction_base_file_path)))

    def test_get_reaction_params(self):

        # testing _get_reaction_params() method
        env = gym.make(ENV_NAME)
        file_path = "../../chemistrylab/reactions/available_reactions/chloro_wurtz.py"

        # unpacking all variables contained in chloro_wurtz reacton file
        reactants = REACTANTS
        products = PRODUCTS
        solutes = SOLUTES
        desired = DESIRED
        Ti_ = Ti
        Vi_ = Vi
        Tmin_ = Tmin
        Tmax_ = Tmax
        Vmin_ = Vmin
        Vmax_ = Vmax
        dt_ = dt
        dT_ = dT
        dV_ = dV
        activ_energy_arr_ = activ_energy_arr
        stoich_coeff_arr_ = stoich_coeff_arr
        conc_coeff_arr_ = conc_coeff_arr

        # using _get_reaction_params method
        reaction_params = env.reaction._get_reaction_params(reaction_filepath=file_path)

        # checking if the _get_reaction_params method gets the correct values
        self.assertEqual(reactants, reaction_params["REACTANTS"])
        self.assertEqual(products, reaction_params["PRODUCTS"])
        self.assertEqual(solutes, reaction_params["SOLUTES"])
        self.assertEqual(desired, reaction_params["DESIRED"])
        self.assertEqual(Ti_, reaction_params["Ti"])
        self.assertEqual(Vi_, reaction_params["Vi"])
        self.assertEqual(Tmin_, reaction_params["Tmin"])
        self.assertEqual(Tmax_, reaction_params["Tmax"])
        self.assertEqual(Vmin_, reaction_params["Vmin"])
        self.assertEqual(Vmax_, reaction_params["Vmax"])
        self.assertEqual(dt_, reaction_params["dt"])
        self.assertEqual(dT_, reaction_params["dT"])
        self.assertEqual(dV_, reaction_params["dV"])
        self.assertEqual(activ_energy_arr_.tolist(), reaction_params["activ_energy_arr"].tolist())
        self.assertEqual(stoich_coeff_arr_.tolist(), reaction_params["stoich_coeff_arr"].tolist())
        self.assertEqual(conc_coeff_arr_.tolist(), reaction_params["conc_coeff_arr"].tolist())

    def test_reset(self):

        # testing reset() method
        env = gym.make(ENV_NAME)
        env.reset()

        # tests to see if reset the vessel parameters to original values specified in reaction file
        self.assertEqual(env.vessels.v_min, Vmin)
        self.assertEqual(env.vessels.v_max, Vmax)
        self.assertEqual(env.vessels.volume, Vi)
        self.assertEqual(env.vessels.temperature, Ti)
        self.assertEqual(env.vessels.Tmin, Tmin)
        self.assertEqual(env.vessels.Tmax, Tmax)
        self.assertEqual(env.vessels.default_dt, dt)

    def test_plotting_step(self):

        # testing test_plotting_step() method to see if they return correct plotting data arrays
        env = gym.make(ENV_NAME)
        env.reset()
        env.render(mode='human')

        # actions that engine will perform
        action = np.ones(env.action_space.shape[0])
        env.step(action)

        # chaning plot_data lists to work with assert
        plot_data_state = list(str(env.plot_data_state).replace('[','').replace(']','').split(","))
        plot_data_state = [float(state) for state in plot_data_state]
        plot_data_mol = list(str(env.plot_data_mol).replace('[', '').replace(']', '').split(","))
        plot_data_mol = [float(state) for state in plot_data_mol]

        # getting state array for amount of mols pertaining to each material
        state_mols = np.zeros(
            4 +  # time T V P
            len(env.reaction.materials),
            dtype=np.float32
        )
        materials = REACTANTS + PRODUCTS
        for i in range(len(materials)):
            state_mols[i + 4] = env.reaction.n[i]

        # test if plot_data_state is the correct value
        np.testing.assert_almost_equal(plot_data_state, env.state[:4].tolist(), decimal=3)
        np.testing.assert_almost_equal(plot_data_mol, state_mols[4:].tolist(), decimal=3)

    def test_temperature_increase(self):

        # testing perform action for temperature increase
        env = gym.make(ENV_NAME)
        env.reset()

        # setting action to increase temperature
        action = np.zeros(env.action_space.shape[0])
        action[0] = 1
        env.step(action)

        desired_temp = env.reaction.Ti + env.reaction.dT
        actual_temp = env.vessels.get_temperature()
        self.assertAlmostEqual(desired_temp, actual_temp, places=6)

    def test_volume_increase(self):

        # testing perform action for volume increase
        env = gym.make(ENV_NAME)
        env.reset()

        # setting action to increase volume
        action = np.zeros(env.action_space.shape[0])
        action[1] = 1
        env.step(action)

        desired_volume = env.reaction.Vi + env.reaction.dV
        actual_volume = env.vessels.get_volume()
        self.assertAlmostEqual(desired_volume, actual_volume, places=6)

    def test_temperature_decrease(self):

        # testing perform action for temperature decrease
        env = gym.make(ENV_NAME)
        env.reset()

        # setting action to decrease temperature
        action = np.zeros(env.action_space.shape[0])
        action[0] = 0
        env.step(action)

        desired_temp = max((env.reaction.Ti - env.reaction.dT), env.vessels.Tmin)
        actual_temp = env.vessels.get_temperature()
        self.assertAlmostEqual(desired_temp, actual_temp, places=6)

    def test_volume_decrease(self):

        # testing perform action for volume decrease
        env = gym.make(ENV_NAME)
        env.reset()

        # setting action to decrease volume
        action = np.zeros(env.action_space.shape[0])
        action[1] = 0
        env.step(action)

        desired_volume = max((env.reaction.Vi - env.reaction.dV), env.vessels.v_min)
        actual_volume = env.vessels.get_volume()
        self.assertAlmostEqual(desired_volume, actual_volume, places=6)

    def test_add_nacl(self):

        # test adding NaCl
        env = gym.make(ENV_NAME)
        env.reset()
        action = np.zeros(env.action_space.shape[0])

        # checking to see if NaCl is in current material_dict before reaction goes through
        self.assertNotIn('NaCl', env.vessels.get_material_dict())

        # stepping through the action and checking to see if NaCl is in material_dict as it should be
        env.step(action)
        self.assertIn('NaCl', env.vessels.get_material_dict())

    def test_solvers(self):
        solvers = {'newton', 'RK45', 'RK23', 'DOP853', 'BDF', 'LSODA'}

        for i in range(1):
            for solver in solvers:
                print(solver)
                env = ReactionBenchEnv_0()
                env.reaction.solver = solver
                done = False
                state = env.reset()
                round = 0
                while not done:
                    action = np.zeros(env.action_space.shape[0])
                    if round == 0:
                        action = np.ones(env.action_space.shape[0])
                    else:
                        action[0] = 1
                        action[1] = 1
                    state, reward, done, _ = env.step(action)
                    if round == 20:
                        done = True
                    round += 1




if __name__ == '__main__':
    unittest.main()