import numpy as np
import gym
import gym.spaces
import matplotlib.pyplot as plt
from chemistrylab.reactions.reaction_0 import Reaction

R = 0.008314462618 # Gas constant (kPa * m**3 * mol**-1 * K**-1)
wave_max = 800
wave_min = 200

class ODEWorldEnv(gym.Env):
    def __init__(
        self,
        reaction=Reaction,
        n_steps=50,
        dt=0.01,
        Ti=300,
        Tmin=250,
        Tmax=500,
        dT=50, 
        Vi = 0.002,
        Vmin = 0.001,
        Vmax = 0.005,
        dV = 0.0005,
        overlap = False,
        ):
        self.n_steps = n_steps # Time steps per action
        self.dt = dt # Time step of reaction (s)
        self.tmax = 10.0 # Maximum time (s) (n_steps * dt * max_episode steps)
        self.Ti = Ti # Initial temperature (K)
        self.Tmin = Tmin # Minimum temperature of the system (K)
        self.Tmax = Tmax # Maximum temperature of the system (K)
        self.dT = dT # Maximum change in temperature per action (K)

        self.Vi = Vi # Initial Volume (m**3)
        self.Vmin = Vmin # Minimum Volume of the system (m**3)
        self.Vmax = Vmax # Maximum Volume of the system (m**3)
        self.dV = dV # Maximum change in Volume per action (m**3)
        self.reaction = reaction(overlap = overlap)
        self.Pmax = self.reaction.max_mol * R * Tmax / Vmin
        # Defining action and observation space for OpenAI Gym framework
        act_low = np.zeros(self.reaction.initial_in_hand.shape[0]+2, dtype=np.float32) # 2 means change of T and V
        act_high = np.ones(self.reaction.initial_in_hand.shape[0]+2, dtype=np.float32) # shape[0] means change reactant
        self.action_space = gym.spaces.Box(low=act_low, high=act_high)
        absorb = self.reaction.get_spectra(Vi)
        # 4 mean time T V P
        # ni.shape[0] mean reactant
        # absorb.shape[0] man the spectra
        obs_low = np.zeros(self.reaction.initial_in_hand.shape[0]+4+absorb.shape[0], dtype=np.float32)
        obs_high = np.ones(self.reaction.initial_in_hand.shape[0]+4+absorb.shape[0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high)

        # Reset the environment to start
        self.reset()

    # Update the state vector with current time, temperature, and amount of reagents added
    def _update_state(self):
        absorb = self.reaction.get_spectra(self.V)
        self.state = np.zeros(
                            4 + # time T V P
                            self.reaction.initial_in_hand.shape[0] + # reactant
                            absorb.shape[0], # spectra
                            dtype=np.float32,
                            )
        self.state[0] = self.t / self.tmax # time
        self.state[1] = (self.T - self.Tmin) / (self.Tmax - self.Tmin) # T
        self.state[2] = (self.V - self.Vmin) / (self.Vmax - self.Vmin) # V
        self.state[3] = self.reaction.get_total_pressure(self.V, self.T) / self.Pmax # total pressure
        for i in range(self.reaction.initial_in_hand.shape[0]):
            self.state[i+4] = self.reaction.n[i] / self.reaction.nmax[i]
        for i in range(absorb.shape[0]):
            self.state[i + self.reaction.initial_in_hand.shape[0] + 4] = absorb[i]

    # Reinitializes the environment
    def reset(self):
        self._first_render = True
        self.done = False
        self.t = 0.0
        self.T = 1.0 * self.Ti
        self.V = 1.0 * self.Vi
        self.reaction.reset() # reinitialize the reaction class
        self.plot_data_state = [
                                    [0.0],
                                    [(self.Ti - self.Tmin) / (self.Tmax - self.Tmin)],
                                    [(self.Vi - self.Vmin) / (self.Vmax - self.Vmin)],
                                    [self.reaction.get_total_pressure(self.V, self.T)]
                                ]
        self.plot_data_mol = []
        self.plot_data_concentration = []
        # [0] is time, [1] is Tempurature, [2:] is each species
        C = self.reaction.get_concentration(self.V)
        for i in range(self.reaction.nmax.shape[0]):
            self.plot_data_mol.append([self.reaction.n[i]])
            self.plot_data_concentration.append([C[i]])

        self._update_state()
        return self.state

    # Updating the environment with the agents action
    def step(self, action):
        reward = 0.0
        for i in range(self.reaction.cur_in_hand.shape[0]):
            # Actions [1:] add reactants to system, 0.0 = 0%, 1.0 = 100%
            # If action [i] is larger than the amount of reactant i, all is added
            if action[i+2] > self.reaction.cur_in_hand[i]:
                # in case over use we apply a penalty
                reward -= 0.1 * (action[i+2] - self.reaction.cur_in_hand[i])
            dn = np.min([action[i+2], self.reaction.cur_in_hand[i]])
            self.reaction.n[i] += dn
            self.reaction.cur_in_hand[i] -= dn
        for i in range(self.n_steps):
            # Action [0] changes temperature by, 0.0 = -dT K, 0.5 = 0 K, 1.0 = +dT K
            # Temperature change is linear between frames
            self.T += 2.0 * self.dT * (action[0] - 0.5) / self.n_steps
            self.T = np.min([np.max([self.T, self.Tmin]), self.Tmax])
            self.V += 2.0 * self.dV * (action[1] - 0.5) / self.n_steps
            self.V = np.min([np.max([self.V, self.Vmin]), self.Vmax])
            d_reward = self.reaction.update(self.T, self.V, self.dt)
            self.t += self.dt # Increase time by time step
            reward += d_reward # Reward is change in concentration of C
            # Record data for render mode
            self.plot_data_state[0].append(self.t)
            self.plot_data_state[1].append( (self.T - self.Tmin) / (self.Tmax - self.Tmin) )
            self.plot_data_state[2].append( (self.V - self.Vmin) / (self.Vmax - self.Vmin) )
            self.plot_data_state[3].append(self.reaction.get_total_pressure(self.V, self.T))
            C = self.reaction.get_concentration(self.V)
            for i in range(self.reaction.n.shape[0]):
                self.plot_data_mol[i].append(self.reaction.n[i])
                self.plot_data_concentration[i].append(C[i])

        self._update_state()

        return self.state, reward, self.done, {}

    def render(self, model='human'):
        if model == 'human':
            self.human_render()
        elif model == 'full':
            self.full_render()

    def human_render(self, mode='plot'):
        # Wavelength array for plotting
        spectra_len = self.state.shape[0] - 4 - self.reaction.initial_in_hand.shape[0]
        absorb = self.state[4 + self.reaction.initial_in_hand.shape[0]:]
        wave = np.linspace(wave_min, wave_max, spectra_len, endpoint=True, dtype=np.float32)
        num_list = [self.state[0], self.state[1], self.state[2], self.state[3]] + self.reaction.get_ni_num()
        label_list = ['t', 'T', 'V','P'] + self.reaction.get_ni_label()
        if self._first_render:
            plt.close('all')
            plt.ion()
            self._plot_fig, self._plot_axs = plt.subplots(1, 2, figsize=(12, 6))
            # spectra
            self._plot_line1 = self._plot_axs[0].plot(
                                                    wave, # wave range from wave_min to wave_max
                                                    absorb, # absorb spectra
                                                    )[0]
            self._plot_axs[0].set_xlim([wave_min, wave_max])
            self._plot_axs[0].set_ylim([0, 1.2])
            self._plot_axs[0].set_xlabel('Wavelength (nm)')
            self._plot_axs[0].set_ylabel('Absorbance')
            # bar chart for time Tempuratue pressure amount of species left in hand
            self._plot_axs[1].bar(
                                range(len(num_list)),
                                num_list,
                                tick_label=label_list,
                                )[0]
            self._plot_axs[1].set_ylim([0, 1])
            # draw the graph
            self._plot_fig.canvas.draw()
            plt.show()
            self._first_render = False
        else:
            self._plot_line1.set_xdata(wave)
            self._plot_line1.set_ydata(absorb)
            # bar chart
            self._plot_axs[1].cla()
            self._plot_axs[1].bar(
                                range(len(num_list)),
                                num_list,
                                tick_label=label_list,
                                )[0]
            self._plot_axs[1].set_ylim([0, 1])
            # draw the graph
            self._plot_fig.canvas.draw()
            plt.pause(0.000001)

    # Rendering the environment for a human to visualize the state
    def full_render(self, mode='plot'):
        spectra_len = self.state.shape[0] - 4 - self.reaction.initial_in_hand.shape[0]
        absorb = self.state[4 + self.reaction.initial_in_hand.shape[0]:]
        wave = np.linspace(wave_min, wave_max, spectra_len, endpoint=True, dtype=np.float32)
        num_list = [self.state[0], self.state[1], self.state[2], self.state[3]] + self.reaction.get_ni_num()
        label_list = ['t', 'T', 'V','P'] + self.reaction.get_ni_label()
        peak = self.reaction.get_spectra_peak(self.V)
        dash_spectra = self.reaction.get_dash_line_spectra(self.V)
        # First render initializes the figure
        if self._first_render:
            plt.close('all')
            plt.ion()
            self._plot_fig, self._plot_axs = plt.subplots(2, 3, figsize=(24, 12))
            # t-species_amount graph ********************** (0, 0)
            self._plot_lines_amount = []
            for i in range(self.reaction.n.shape[0]):
                self._plot_lines_amount.append(
                                    self._plot_axs[0, 0].plot(
                                        self.plot_data_state[0], # time 
                                        self.plot_data_mol[i], # mol of species C[i]
                                        label = self.reaction.labels[i], # name of species C[i]
                                        )[0],
                                    )
            self._plot_axs[0, 0].set_xlim([0.0, self.dt*self.n_steps])
            self._plot_axs[0, 0].set_ylim([0.0, np.max(self.reaction.nmax)])
            self._plot_axs[0, 0].set_xlabel('Time (s)')
            self._plot_axs[0, 0].set_ylabel('amount (mol)')
            self._plot_axs[0, 0].legend()
            # t-species_CONCENTRATION graph ********************** (0, 1)
            self._plot_lines_concentration = []
            for i in range(self.reaction.n.shape[0]):
                self._plot_lines_concentration.append(
                                    self._plot_axs[0, 1].plot(
                                        self.plot_data_state[0], # time 
                                        self.plot_data_concentration[i], # concentration of species C[i]
                                        label = self.reaction.labels[i], # name of species C[i]
                                        )[0],
                                    )
            self._plot_axs[0, 1].set_xlim([0.0, self.dt*self.n_steps])
            self._plot_axs[0, 1].set_ylim([0.0, np.max(self.reaction.nmax)/ (self.Vmin * 1000)])
            self._plot_axs[0, 1].set_xlabel('Time (s)')
            self._plot_axs[0, 1].set_ylabel('Concentration (mol/L)')
            self._plot_axs[0, 1].legend()
            # T-t + V-t *************** (0, 2)
            self._plot_line_1 = self._plot_axs[0, 2].plot(
                                                    self.plot_data_state[0], # t
                                                    self.plot_data_state[1], # T
                                                    label = 'T',
                                                    )[0]
            self._plot_line_2 = self._plot_axs[0, 2].plot(
                                                    self.plot_data_state[0], # t
                                                    self.plot_data_state[2], # V
                                                    label = 'V',
                                                    )[0]                                       
            self._plot_axs[0, 2].set_xlim([0.0, self.dt*self.n_steps])
            self._plot_axs[0, 2].set_ylim([0, 1])
            self._plot_axs[0, 2].set_xlabel('Time (s)')
            self._plot_axs[0, 2].set_ylabel('T and V (map to range [0, 1])')
            self._plot_axs[0, 2].legend()
            # P-t (1, 0)
            self._plot_line_3 = self._plot_axs[1, 0].plot(
                                                    self.plot_data_state[0], # t
                                                    self.plot_data_state[3], # P
                                                    )[0]
            self._plot_axs[1, 0].set_xlim([0.0, self.dt*self.n_steps])
            self._plot_axs[1, 0].set_ylim([0, self.Pmax])
            self._plot_axs[1, 0].set_xlabel('Time (s)')
            self._plot_axs[1, 0].set_ylabel('Pressure (kPa)')
            # solid spectra (1, 1)
            self._plot_axs[1, 1].plot(
                                    wave, # wave range from wave_min to wave_max
                                    absorb, # absorb spectra
                                    )[0]
            # dash spectra
            for i in range(len(dash_spectra)):
                self._plot_axs[1, 1].plot(wave, dash_spectra[i], linestyle='dashed')
            # label
            for i in range(self.reaction.n.shape[0]):
                self._plot_axs[1, 1].scatter(peak[i][0], peak[i][1], label = peak[i][2])

            self._plot_axs[1, 1].set_xlim([wave_min, wave_max])
            self._plot_axs[1, 1].set_ylim([0, 1.2])
            self._plot_axs[1, 1].set_xlabel('Wavelength (nm)')
            self._plot_axs[1, 1].set_ylabel('Absorbance')
            self._plot_axs[1, 1].legend()
            # bar chart for time Tempuratue pressure amount of species left in hand (1, 2)
            self._plot_axs[1, 2].bar(
                                range(len(num_list)),
                                num_list,
                                tick_label=label_list,
                                )[0]
            self._plot_axs[1, 2].set_ylim([0, 1])
            # draw the graph
            self._plot_fig.canvas.draw()
            plt.show()
            self._first_render = False

        # All other renders update the existing figure with the current state
        else:
            # set data for  t-species_amount graph
            curent_time =  self.plot_data_state[0][-1]
            for i in range(self.reaction.n.shape[0]):
                self._plot_lines_amount[i].set_xdata(self.plot_data_state[0])
                self._plot_lines_amount[i].set_ydata(self.plot_data_mol[i])
                # set data for  t-species_concentration graph ***********
                self._plot_lines_concentration[i].set_xdata(self.plot_data_state[0])
                self._plot_lines_concentration[i].set_ydata(self.plot_data_concentration[i])
            self._plot_axs[0, 0].set_xlim([0.0, curent_time]) # reset xlime since t extend
            self._plot_axs[0, 1].set_xlim([0.0, curent_time]) # reset xlime since t extend
            # set data for t-T + V-t graph ****************
            self._plot_line_1.set_xdata(self.plot_data_state[0])
            self._plot_line_1.set_ydata(self.plot_data_state[1])
            self._plot_line_2.set_xdata(self.plot_data_state[0])
            self._plot_line_2.set_ydata(self.plot_data_state[2])
            self._plot_axs[0, 2].set_xlim([0.0, curent_time]) # reset xlime since t extend
            # set data for t-P graph
            self._plot_line_3.set_xdata(self.plot_data_state[0])
            self._plot_line_3.set_ydata(self.plot_data_state[3])
            self._plot_axs[1, 0].set_xlim([0.0, curent_time]) # reset xlime since t extend
            # solid Spectra
            self._plot_axs[1, 1].cla()
            self._plot_axs[1, 1].plot(
                                    wave, # wave range from wave_min to wave_max
                                    absorb, # absorb spectra
                                    )[0]

            # dash spectra
            for i in range(len(dash_spectra)):
                self._plot_axs[1, 1].plot(wave, dash_spectra[i], linestyle='dashed')
            
            # label
            for i in range(self.reaction.n.shape[0]):
                self._plot_axs[1, 1].scatter(peak[i][0], peak[i][1], label = peak[i][2])

            self._plot_axs[1, 1].set_xlim([wave_min, wave_max])
            self._plot_axs[1, 1].set_ylim([0, 1.2])
            self._plot_axs[1, 1].set_xlabel('Wavelength (nm)')
            self._plot_axs[1, 1].set_ylabel('Absorbance')
            self._plot_axs[1, 1].legend()
            # bar chart
            self._plot_axs[1, 2].cla()
            self._plot_axs[1, 2].bar(
                                range(len(num_list)),
                                num_list,
                                tick_label=label_list,
                                )[0]
            self._plot_axs[1, 2].set_ylim([0, 1])
            # draw the graph
            self._plot_fig.canvas.draw()
            plt.pause(0.000001)
