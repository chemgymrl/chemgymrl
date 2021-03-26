import unittest
import sys
import os
sys.path.append('../../../')
sys.path.append('../../../chemistrylab/reactions')
import gym
from chemistrylab.reactions.available_reactions.chloro_wurtz import *
from chemistrylab.reactions.get_reactions import convert_to_class
import numpy as np

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
        path = "..\\..\\..\\chemistrylab\\reactions\\available_reactions"
        os.chdir(path)
        file_path = path + '\\chloro_wurtz.py'
        reaction_base_file_path = env.reaction._find_reaction_file(reaction_file="chloro_wurtz")

        # checks if file paths are the same
        self.assertEqual(True, bool(os.path.samefile(file_path, reaction_base_file_path)))

    def test_get_reaction_params(self):

        # testing _get_reaction_params() method
        env = gym.make(ENV_NAME)
        file_path = "..\\..\\..\\chemistrylab\\reactions\\available_reactions\\chloro_wurtz.py"

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

    def test_get_concentration(self):

        # testing get_concentration method
        env = gym.make(ENV_NAME)

        Volume = 0.2
        C = np.zeros(env.reaction.n.shape[0], dtype=np.float32)
        for i in range(env.reaction.n.shape[0]):
            C[i] = env.reaction.n[i]/Volume

        C_reaction = env.reaction.get_concentration(Volume)

        self.assertEqual(C.tolist(), C_reaction.tolist())

    def test_get_spectra_absorb_shape(self):

        # testing get_spectra() method by checking if absorb has the correct shape
        env = gym.make(ENV_NAME)

        # checking against correct observation shape (achieved with correct absorb.shape[0]
        obs_low = np.zeros(208, dtype=np.float32)
        obs_high = np.ones(208, dtype=np.float32)

        # gets observation shape via absorb and get_spectra() method
        absorb = env.reaction.get_spectra(env.vessels.get_volume())
        obs_low_reaction = np.zeros(env.reaction.initial_in_hand.shape[0] + 4 + absorb.shape[0],
                                    dtype=np.float32)
        obs_high_reaction = np.ones(env.reaction.initial_in_hand.shape[0] + 4 + absorb.shape[0],
                                    dtype=np.float32)

        # tests to see if the observation shape is correct
        self.assertEqual(obs_low.tolist(), obs_low_reaction.tolist())
        self.assertEqual(obs_high.tolist(), obs_high_reaction.tolist())


    def test_get_spectra_absorb_values(self):
        
        # testing get_spectra() method by checkng if absorb has the correct values
        env = gym.make(ENV_NAME)
        Volume = 0.1

        # convert the volume in litres to the volume in m**3
        Volume = Volume / 1000

        # set the wavelength space
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)

        # define an array to contain absorption data
        absorb = np.zeros(x.shape[0], dtype=np.float32)

        # obtain the concentration array
        C = env.reaction.get_concentration(Volume)

        # iterate through the spectral parameters in self.params and the wavelength space
        for i, item in enumerate(env.reaction.params):
            for j in range(item.shape[0]):
                for k in range(x.shape[0]):
                    amount = C[i]
                    height = item[j, 0]
                    decay_rate = np.exp(
                        -0.5 * (
                                (x[k] - env.reaction.params[i][j, 1]) / env.reaction.params[i][j, 2]
                        ) ** 2.0
                    )
                    if decay_rate < 1e-30:
                        decay_rate = 0
                    absorb[k] += amount * height * decay_rate

        # absorption must be between 0 and 1
        absorb = np.clip(absorb, 0.0, 1.0)

        absorb_reaction = env.reaction.get_spectra(V=0.1)

        self.assertEqual(absorb.tolist(), absorb_reaction.tolist())

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
        np.testing.assert_almost_equal(plot_data_state, env.state[:4].tolist(), decimal=5)
        np.testing.assert_almost_equal(plot_data_mol, state_mols[4:].tolist(), decimal=5)

