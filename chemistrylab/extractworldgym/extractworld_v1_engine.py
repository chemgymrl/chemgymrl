import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
from chemistrylab.extract_algorithms.extractions import water_oil_v1
from chemistrylab.extract_algorithms import separate
import cmocean
from chemistrylab.chem_algorithms import util

# A dictionary of available extractions
extraction_dict = {'water_oil_2': water_oil_v1,
                   }


class ExtractWorldEnv(gym.Env):
    def __init__(self,
                 n_steps=100,
                 dt=0.01,
                 extraction='water_oil_2',
                 n_vessel_pixels=100,
                 max_valve_speed=10,
                 extraction_vessel=None,
                 target_material=None,
                 ):
        self.n_steps = n_steps
        self.dt = dt
        self.n_vessel_pixels = n_vessel_pixels
        self.max_valve_speed = max_valve_speed
        self.extraction_vessel = extraction_vessel
        # print(self.extraction_vessel._solute_dict)
        self.target_material = target_material

        self.extraction = extraction_dict['water_oil_2'].Extraction(extraction_vessel=self.extraction_vessel,
                                                                    n_vessel_pixels=self.n_vessel_pixels,
                                                                    max_valve_speed=self.max_valve_speed,
                                                                    target_material=self.target_material)
        self.vessels = None
        self.external_vessels = None  # oil vessel for example, renamed it to external_vessels for generalization
        self.state = None
        # self.observation_space = self.extraction.get_observation_space()
        self.action_space = self.extraction.get_action_space()
        self.done = False
        self._first_render = True
        self._plot_fig = None
        self._plot_axs = None
        # Reset the environment to start
        self.reset()

    # Reinitializes the environment
    def reset(self):
        self.done = False
        self._first_render = True
        self.vessels, self.external_vessels, self.state = self.extraction.reset(
            extraction_vessel=self.extraction_vessel)
        return self.state

    # Updating the environment with the agents action
    def step(self, action):
        self.vessels, self.external_vessels, reward, self.done = self.extraction.perform_action(vessels=self.vessels,
                                                                                                oil_vessel=self.external_vessels,
                                                                                                action=action)
        self.state = util.generate_state(self.vessels, max_n_vessel=self.extraction.n_total_vessels)
        self.n_steps -= 1
        if self.n_steps == 0:
            self.done = True
            reward = self.extraction.done_reward(self.vessels[2])
        return self.state, reward, self.done, {}

    def render(self, model='human'):
        if model == 'human':
            self.human_render()

    def human_render(self, mode='plot'):
        Ps = []
        for Vessel in self.vessels:
            p, v = Vessel.get_position_and_variance()
            Ps.append(np.zeros((len(p), separate.x.shape[0]), dtype=np.float32))
        for i in range(len(Ps)):
            p, v = self.vessels[i].get_position_and_variance(dict_or_list='dict')
            t = -1.0 * np.log(v * np.sqrt(2.0 * np.pi))
            j = 0
            for layer in p:
                # Loop over each phase
                for k in range(Ps[i].shape[1]):
                    # Calculate the value of the Gaussian for that phase and x position
                    Ps[i][j, k] = self.vessels[i].get_material_volume(layer) * np.exp(
                        -1.0 * (((separate.x[k] - p[layer]) / (2.0 * v)) ** 2) + t)
                j += 1
        Ls = np.reshape(np.array([x[2] for x in self.state]),
                        (self.extraction.n_total_vessels, self.extraction.n_vessel_pixels, 1))
        if self._first_render:
            plt.close('all')
            plt.ion()
            self._plot_fig, self._plot_axs = plt.subplots(self.extraction.n_total_vessels, 2, figsize=(12, 6),
                                                          gridspec_kw={'width_ratios': [3, 1]})
            for i in range(len(Ps)):
                p, v = self.vessels[i].get_position_and_variance()
                j = 0
                for layer in p:
                    if layer == 'Air':
                        color = self.vessels[i].Air.get_color()
                    else:
                        color = self.vessels[i].get_material_color(layer)
                    color = cmocean.cm.delta(color)
                    self._plot_axs[i, 0].plot(separate.x, Ps[i][j], label=layer, c=color)
                    j += 1
                self._plot_axs[i, 0].set_xlim([separate.x[0], separate.x[-1]])
                self._plot_axs[i, 0].set_ylim([0, self.vessels[i].v_max])
                self._plot_axs[i, 0].set_xlabel('Separation')
                self._plot_axs[i, 0].legend()
                mappable = self._plot_axs[i, 1].pcolormesh(Ls[i], vmin=0, vmax=1, cmap=cmocean.cm.delta)
                self._plot_axs[i, 1].set_xticks([])
                self._plot_axs[i, 1].set_ylabel('Height')
                # self._plot_axs[i, 1].colorbar(mappable)
                self._plot_fig.canvas.draw()
                plt.show()
                self._first_render = False
        else:
            for i in range(len(Ps)):
                self._plot_axs[i, 0].clear()
                p, v = self.vessels[i].get_position_and_variance()
                j = 0
                for layer in p:
                    if layer == 'Air':
                        color = self.vessels[i].Air.get_color()
                    else:
                        color = self.vessels[i].get_material_color(layer)
                    color = cmocean.cm.delta(color)
                    self._plot_axs[i, 0].plot(separate.x, Ps[i][j], label=layer, c=color)
                    j += 1
                self._plot_axs[i, 0].set_xlim([separate.x[0], separate.x[-1]])
                self._plot_axs[i, 0].set_ylim([0, self.vessels[i].v_max])
                self._plot_axs[i, 0].set_xlabel('Separation')
                self._plot_axs[i, 0].legend()
                mappable = self._plot_axs[i, 1].pcolormesh(Ls[i], vmin=0, vmax=1, cmap=cmocean.cm.delta)
                # self._plot_axs[i, 1].colorbar(mappable)
                self._plot_fig.canvas.draw()
                # plt.show()
                self._first_render = False
